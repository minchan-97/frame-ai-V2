"""
gascore_engine.py — GasCore 통합 프레임워크
============================================
Layer 1: 진화 인덱스 (SOM + LLM + 사용자 검증)
Layer 2: CoreAI v1 가드레일 (NeuralMarkov + RAG)
Layer 3: XAI (토큰별 이탈 설명)

세 레이어가 순환하며 코퍼스가 점점 전문화됨
"""
from __future__ import annotations
import numpy as np
import time
from collections import Counter, defaultdict
from typing import Optional
from dataclasses import dataclass, field

try:
    from neural_markov_engine import NeuralMarkovEngine
    NM_OK = True
except Exception:
    NM_OK = False

try:
    from core_ai_v2_engine import CoreAIv2Engine
    V2_OK = True
except Exception:
    V2_OK = False

try:
    from evolving_engine import EvolvingIndexEngine
    EV_OK = True
except Exception:
    EV_OK = False

_rng = np.random.default_rng(42)

# ── 토크나이저 ────────────────────────────────────────────────
_JOSA = ["에서","에게","으로","부터","까지","와","과","을","를",
         "은","는","이","가","의","도","만","에","로"]
_EOMI = ["했습니다","합니다","됩니다","있습니다","입니다",
         "했다","한다","이다","하고","해서","하여","되어",
         "이며","하는","된","한","이고","이에"]

def tokenize(text: str) -> list:
    tokens = []
    for word in text.replace("\n"," ").split():
        word = word.strip(".,!?()[]\"'~?：:；;")
        stem = word
        for s in sorted(_JOSA+_EOMI, key=len, reverse=True):
            if word.endswith(s) and len(word)>len(s)+1:
                stem = word[:-len(s)]; break
        if stem and len(stem) > 1:
            tokens.append(stem)
    return tokens


def clean_corpus(corpus_text: str) -> str:
    """
    코퍼스 정제
    - URL, 숫자뭉침, 특수문자 덩어리 제거
    - 의미 있는 한국어/영어 문장만 남김
    """
    import re
    lines = corpus_text.split('\n')
    clean = []
    seen  = set()
    for line in lines:
        s = line.strip()
        if not s or len(s) < 6:
            continue
        # URL 제거
        if re.search(r'https?://', s):
            continue
        # 000000 같은 연속 숫자 패턴
        if re.search(r'0{4,}|9{4,}|\d{6,}', s):
            continue
        # ￭ 특수 구분자 포함
        if '￭' in s or '◆' in s or '◇' in s:
            continue
        # 숫자+한글 뭉침 (날짜표 등): 10자 이상 숫자한글혼합
        if re.search(r'\d{3,}[가-힣]{1,3}\d{3,}', s):
            continue
        # 한국어/영어 비율 40% 미만 제거 (강화)
        korean_en = len(re.findall(r'[가-힣a-zA-Z]', s))
        total = len(s.replace(' ',''))
        if total > 0 and korean_en / total < 0.4:
            continue
        # 목차 숫자 패턴
        if re.match(r'^[\d\s]{8,}', s):
            continue
        # 한자/특수기호만
        if re.match(r'^[Ⅰ-Ⅹ①-⑳\s]+$', s):
            continue
        # 단어 평균 길이가 너무 긴 경우 (PDF 뭉침)
        words = s.split()
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len > 15:  # 평균 단어 15자 초과 → 뭉침
                continue
        # 중복 제거
        key = re.sub(r'\s+', '', s)[:40]
        if key in seen:
            continue
        seen.add(key)
        clean.append(s)
    return '\n'.join(clean)


# ── XAI 결과 ─────────────────────────────────────────────────
@dataclass
class XAIResult:
    """XAI 설명 결과"""
    verdict: str                    # PASS/WARNING/FATAL
    avg_logp: float                 # 전체 평균 logP
    token_scores: list = field(default_factory=list)   # [(토큰, logP, 이상여부)]
    outlier_tokens: list = field(default_factory=list) # 이탈 토큰들
    cluster_hint: str = ""          # 가장 가까운 SOM 클러스터
    explanation: str = ""           # 한국어 설명
    ms: float = 0.0


# ── Layer 3: XAI ─────────────────────────────────────────────
class XAILayer:
    """
    토큰별 logP 추적 + SOM 클러스터 연결
    "왜 이 답변이 이탈로 판정됐는가"를 설명
    """
    def __init__(self, nm: NeuralMarkovEngine,
                 som_neurons: Optional[dict] = None):
        self.nm = nm
        self.som_neurons = som_neurons or {}  # 뉴런 → 문장 리스트

    def explain(self, text: str, logp_thr: float = -11.5) -> XAIResult:
        t0 = time.perf_counter()
        tokens = tokenize(text)
        if not tokens or not self.nm or not self.nm.is_trained:
            return XAIResult(verdict="SKIP", avg_logp=0.0)

        # 토큰별 logP 계산
        token_scores = []
        total_lp = 0.0
        scored = 0
        V = len(self.nm.uni)

        for i in range(len(tokens)):
            wc = tokens[i]
            p1 = (self.nm.uni[wc]+self.nm.alpha)/(self.nm.total+self.nm.alpha*V)
            p2 = p3 = 0.0
            if i>=1:
                wp = tokens[i-1]
                p2 = self.nm.bi[wp][wc]/self.nm.uni[wp] if self.nm.uni[wp]>0 else 0
            if i>=2:
                wpp = tokens[i-2]
                p3 = self.nm.tri[(wpp,wp)][wc]/self.nm.bi[wpp][wp] if self.nm.bi[wpp][wp]>0 else 0
            pjm = 0.6*p3+0.3*p2+0.1*p1
            lp = float(np.log(pjm+1e-12))
            is_outlier = lp < -12.0 and self.nm.bi.get(tokens[i-1] if i>0 else "", {}).get(wc,0)==0
            token_scores.append((wc, round(lp,2), is_outlier))
            total_lp += lp; scored += 1

        avg_logp = total_lp/max(scored,1)
        outliers = [t for t,lp,out in token_scores if out]

        # 판정
        if avg_logp >= -10.0:      verdict = "PASS"
        elif avg_logp >= logp_thr: verdict = "WARNING"
        else:                      verdict = "FATAL"

        # SOM 클러스터 힌트
        cluster_hint = ""
        if self.som_neurons:
            # 이탈 토큰과 가장 관련 있는 클러스터 찾기
            for neuron_idx, sents in self.som_neurons.items():
                for s in sents:
                    if any(ot in s for ot in outliers[:3]):
                        cluster_hint = s[:40]
                        break
                if cluster_hint: break

        # 한국어 설명 생성
        explanation = self._make_explanation(
            verdict, avg_logp, outliers, cluster_hint, logp_thr
        )

        ms = (time.perf_counter()-t0)*1000
        return XAIResult(
            verdict=verdict,
            avg_logp=avg_logp,
            token_scores=token_scores,
            outlier_tokens=outliers,
            cluster_hint=cluster_hint,
            explanation=explanation,
            ms=ms,
        )

    def _make_explanation(self, verdict, avg_logp,
                          outliers, cluster_hint, thr) -> str:
        if verdict == "PASS":
            return f"✅ 도메인 안 — 모든 토큰이 학습된 패턴 안에 있어요 (avg_logP: {avg_logp:+.2f})"
        elif verdict == "WARNING":
            out_str = f"'{', '.join(outliers[:3])}'" if outliers else "일부 표현"
            hint = f" → 가장 가까운 개념: '{cluster_hint}'" if cluster_hint else ""
            return (f"🟡 경계 수준 — {out_str}이 코퍼스 경계에 있어요 "
                    f"(avg_logP: {avg_logp:+.2f}, 임계값: {thr}){hint}")
        else:
            out_str = f"'{', '.join(outliers[:3])}'" if outliers else "다수 표현"
            hint = f" → 관련 코퍼스 개념: '{cluster_hint}'" if cluster_hint else ""
            return (f"🔴 도메인 이탈 — {out_str}이 학습된 패턴에서 크게 벗어났어요 "
                    f"(avg_logP: {avg_logp:+.2f}, 임계값: {thr}){hint}")


# ── Layer 2: CoreAI 가드레일 ──────────────────────────────────
@dataclass
class GuardrailResult:
    """가드레일 판정 결과"""
    answer: str
    status: str
    attempts: int
    final_logp: float
    xai: Optional[XAIResult] = None
    history: list = field(default_factory=list)
    total_ms: float = 0.0


class CoreAILayer:
    """
    NeuralMarkov 기반 가드레일
    이탈 시 재생성 루프 + XAI 설명
    """
    def __init__(self, nm: NeuralMarkovEngine,
                 xai: Optional[XAILayer] = None):
        self.nm = nm
        self.xai = xai

    def run(self, question: str, llm_fn,
            max_attempts: int = 3,
            logp_thr: float = -11.5,
            guideline_hint: str = "") -> GuardrailResult:
        t0 = time.perf_counter()
        history = []

        for attempt in range(1, max_attempts+1):
            # LLM 호출
            if attempt == 1:
                prompt = question
            else:
                prompt = (
                    f"이전 답변이 가이드라인에서 벗어났어요.\n"
                    f"가이드라인 참고: {guideline_hint[:300]}\n\n"
                    f"다시 답변해주세요: {question}"
                )

            try:
                answer = llm_fn(prompt)
            except Exception as e:
                answer = f"[LLM 오류: {e}]"
                break

            # 가드레일 평가
            if self.nm and self.nm.is_trained:
                result = self.nm.evaluate(answer, logp_thr=logp_thr)
                status  = result.get("status","FATAL")
                avg_logp = result.get("avg_logp",0.0)
            else:
                status = "PASS"; avg_logp = 0.0

            history.append({
                "attempt": attempt,
                "status":  status,
                "avg_logp": avg_logp,
                "answer_preview": answer[:80],
            })

            if status == "PASS":
                break

        # XAI 설명
        xai_result = None
        if self.xai:
            xai_result = self.xai.explain(answer, logp_thr=logp_thr)

        total_ms = (time.perf_counter()-t0)*1000
        return GuardrailResult(
            answer=answer,
            status=status,
            attempts=attempt,
            final_logp=avg_logp,
            xai=xai_result,
            history=history,
            total_ms=total_ms,
        )


# ── GasCore 통합 프레임워크 ───────────────────────────────────
class GasCoreFramework:
    """
    GasCore 통합 프레임워크
    세 레이어를 순환하며 코퍼스가 전문화됨

    Loop:
      1. 진화 인덱스 → 코퍼스 성장
      2. CoreAI → 성장한 코퍼스로 재학습
      3. XAI → 이탈 설명
      4. 사용자 피드백 → 1번으로
    """
    def __init__(self):
        # Layer 1
        self.evolving = EvolvingIndexEngine() if EV_OK else None
        # Layer 2 — NM (단일) + CoreAI v2 (클러스터)
        self.nm     = NeuralMarkovEngine() if NM_OK else None
        self.coreai_v2 = CoreAIv2Engine() if V2_OK else None
        # Layer 3
        self.xai_layer: Optional[XAILayer] = None
        self.coreai:    Optional[CoreAILayer] = None
        # 검색 엔진
        self.searcher: Optional[CorpusSearcher] = None

        self.corpus_text: str = ""
        self.guideline_hint: str = ""
        self.is_initialized: bool = False
        self.cycle: int = 0

    # ── 초기화 ───────────────────────────────────────────────
    def initialize(self, corpus_text: str,
                   epochs: int = 10,
                   som_grid: int = 6,
                   on_progress=None):
        """전체 프레임워크 초기화"""
        # 코퍼스 자동 정제
        corpus_text = clean_corpus(corpus_text)
        self.corpus_text    = corpus_text
        self.guideline_hint = corpus_text[:1000]

        # Layer 1: 진화 인덱스
        if on_progress: on_progress(10, "진화 인덱스 초기화...")
        if self.evolving:
            self.evolving = EvolvingIndexEngine(grid=som_grid)
            self.evolving.initialize(corpus_text, epochs=epochs)

        # Layer 2: NeuralMarkov (단일)
        if on_progress: on_progress(70, "CoreAI 학습...")
        if self.nm:
            self.nm.train(corpus_text, embedding_dim=32, epochs=epochs)

        # Layer 2+: CoreAI v2 (Hopfield 클러스터 NM)
        if on_progress: on_progress(82, "클러스터 NM 학습...")
        if self.coreai_v2:
            self.coreai_v2.train(corpus_text)

        # Layer 3: XAI
        if on_progress: on_progress(90, "XAI 설정...")
        self._rebuild_layers()

        # 검색 엔진 빌드
        self.searcher = CorpusSearcher()
        self.searcher.build(corpus_text)

        if on_progress: on_progress(100, "완료")
        self.is_initialized = True

    def _rebuild_layers(self):
        """nm 재학습 후 XAI + CoreAI 레이어 재구성"""
        som_neurons = {}
        if self.evolving and self.evolving.som:
            som_neurons = self.evolving.som.neuron_sentences

        if self.nm:
            self.xai_layer = XAILayer(self.nm, som_neurons)
            self.coreai    = CoreAILayer(self.nm, self.xai_layer)

    # ── Layer 1: 진화 실행 ────────────────────────────────────
    def generate_candidates(self, max_candidates: int = 10,
                            logp_thr: float = -13.0,
                            api_key: str = "",
                            model: str = "gpt-4o-mini") -> list:
        if not self.evolving: return []
        return self.evolving.generate_candidates(
            max_candidates=max_candidates,
            logp_thr=logp_thr,
            api_key=api_key,
            model=model,
        )

    def approve_candidate(self, candidate: dict):
        """승인 → 인덱스 추가 → NeuralMarkov 증분 학습 → XAI 재구성"""
        if not self.evolving: return
        self.evolving.user_approve(candidate)
        # NeuralMarkov 증분 학습
        if self.nm:
            sent = candidate["sentence"]
            toks = tokenize(sent)
            for i,t in enumerate(toks):
                self.nm.uni[t]   += 1
                self.nm.total    += 1
                if i>=1: self.nm.bi[toks[i-1]][t]              += 1
                if i>=2: self.nm.tri[(toks[i-2],toks[i-1])][t] += 1
            self._rebuild_layers()

    def reject_candidate(self, candidate: dict, reason: str = ""):
        if not self.evolving: return
        self.evolving.user_reject(candidate, reason)

    # ── Layer 2: CoreAI 가드레일 실행 ────────────────────────
    def run_guardrail(self, question: str, llm_fn,
                      max_attempts: int = 3,
                      logp_thr: float = -11.5) -> GuardrailResult:
        if not self.coreai:
            # 가드레일 없이 LLM만
            try:
                answer = llm_fn(question)
            except Exception as e:
                answer = str(e)
            return GuardrailResult(
                answer=answer, status="SKIP",
                attempts=1, final_logp=0.0,
            )
        return self.coreai.run(
            question=question,
            llm_fn=llm_fn,
            max_attempts=max_attempts,
            logp_thr=logp_thr,
            guideline_hint=self.guideline_hint,
        )

    # ── Layer 3: XAI 독립 실행 ───────────────────────────────
    def explain(self, text: str,
                logp_thr: float = -11.5) -> Optional[XAIResult]:
        """
        가드레일 판정:
          v2 클러스터 NM (우선) + v1 단일 NM (보조)
          둘 다 있으면 더 엄격한 판정 사용
        """
        # v2 클러스터 NM
        v2_result = None
        if self.coreai_v2 and self.coreai_v2.is_trained:
            v2_result = self.coreai_v2.evaluate(text)

        # v1 단일 NM
        v1_result = None
        if self.xai_layer:
            v1_result = self.xai_layer.explain(text, logp_thr)

        # 결합: v2 있으면 우선 사용
        if v2_result and v1_result:
            # 둘 중 더 나쁜 판정 채택 (보수적)
            order = {"PASS":0,"WARNING":1,"CRITICAL":2,"FATAL":3,"SKIP":-1}
            v2_st = v2_result.get("verdict","SKIP")
            v1_st = v1_result.verdict if v1_result else "SKIP"
            if order.get(v2_st,0) >= order.get(v1_st,0):
                # v2 결과를 XAIResult로 변환
                if v1_result:
                    v1_result.verdict    = v2_st
                    v1_result.avg_logp   = v2_result.get("logp", v1_result.avg_logp)
                    v1_result.explanation= (
                        f"클러스터({v2_result.get('cluster','?')}) {v2_st} | "
                        f"logP:{v2_result.get('logp',0):+.2f}")
            return v1_result
        elif v2_result and not v1_result:
            # v2만 있으면 XAIResult 생성
            from dataclasses import dataclass
            st = v2_result.get("verdict","SKIP")
            return XAIResult(
                verdict=st,
                avg_logp=v2_result.get("logp",0),
                explanation=(
                    f"클러스터({v2_result.get('cluster','?')}) | "
                    f"logP:{v2_result.get('logp',0):+.2f}"),
                token_scores=[],
                outlier_tokens=[],
                ms=v2_result.get("ms",0),
            )
        elif v1_result:
            return v1_result
        return None

    # ── 순환 완료 (사용자 피드백 후 재학습) ──────────────────
    def complete_cycle(self, new_corpus_texts: list = None,
                       epochs: int = 5):
        """
        한 순환 완료:
        원본 코퍼스 + 승인된 생성 문장들로 NeuralMarkov 재학습
        """
        # 원본 코퍼스 + 새 문장 합치기
        parts = []
        if self.corpus_text:
            parts.append(self.corpus_text)
        if self.evolving and self.evolving.generated:
            new_sents = [g["sentence"] for g in self.evolving.generated]
            parts.append("\n".join(new_sents))
        if new_corpus_texts:
            parts.extend(new_corpus_texts)

        combined = "\n".join(p for p in parts if p.strip())

        if self.nm and combined.strip():
            self.nm.train(combined, embedding_dim=32, epochs=epochs)

        # 캘리브레이션
        if self.nm and self.corpus_text:
            self.nm._calibrate(self.corpus_text)

        self._rebuild_layers()
        self.cycle += 1

    # ── 저장/로드 ─────────────────────────────────────────────
    def to_dict(self) -> dict:
        try:
            data = {
                "corpus_text":    getattr(self,'corpus_text',''),
                "guideline_hint": getattr(self,'guideline_hint',''),
                "cycle":          getattr(self,'cycle',0),
                "is_initialized": getattr(self,'is_initialized',False),
            }
            ev = getattr(self,'evolving',None)
            if ev:
                try: data["evolving"] = ev.to_dict()
                except Exception: pass
            nm = getattr(self,'nm',None)
            if nm and getattr(nm,'is_trained',False):
                try:
                    data["nm"] = {
                        "uni":   dict(nm.uni),
                        "bi":    {k:dict(v) for k,v in nm.bi.items()},
                        "tri":   {k:dict(v) for k,v in nm.tri.items()},
                        "total": nm.total,
                        "alpha": getattr(nm,"alpha",0.001),
                        "mu":    getattr(nm,"mu",0.0),
                        "std":   getattr(nm,"std",1.0),
                    }
                except Exception: pass
            # CoreAI v2
            try:
                v2 = getattr(self,'coreai_v2',None)
                if v2 is not None and getattr(v2,'is_trained',False) is True:
                    data["coreai_v2"] = {
                        "n_clusters":        v2.n_clusters,
                        "global_vocab":      v2.global_vocab,
                        "corpus_name":       getattr(v2,'corpus_name',''),
                        "train_stats":       getattr(v2,'train_stats',{}),
                        "emb_emb":           v2.embedder.emb if v2.embedder else None,
                        "emb_vocab":         v2.embedder.vocab if v2.embedder else None,
                        "emb_dim":           v2.embedder.dim if v2.embedder else 32,
                        "cluster_sentences": dict(v2.decomposer.cluster_sentences),
                        "cluster_tokens":    dict(v2.decomposer.cluster_tokens),
                        "cluster_keywords":  v2.decomposer.cluster_keywords,
                        "decomp_vocab":      v2.decomposer.vocab,
                        "decomp_W":          v2.decomposer.W,
                        "markovs": {
                            k: {"uni":dict(m.uni),
                                "bi": {k2:dict(vv) for k2,vv in m.bi.items()},
                                "tri":{k2:dict(vv) for k2,vv in m.tri.items()},
                                "total":m.total}
                            for k,m in v2.markovs.items()
                        },
                    }
            except Exception: pass
            return data
        except Exception:
            return {
                "corpus_text":    getattr(self,'corpus_text',''),
                "guideline_hint": getattr(self,'guideline_hint',''),
                "cycle":          getattr(self,'cycle',0),
                "is_initialized": getattr(self,'is_initialized',False),
            }

    @classmethod
    def from_dict(cls, data: dict) -> "GasCoreFramework":
        fw = cls()
        fw.corpus_text    = data.get("corpus_text","")
        fw.guideline_hint = data.get("guideline_hint","")
        fw.cycle          = data.get("cycle",0)
        fw.is_initialized = data.get("is_initialized",False)

        # evolving
        if "evolving" in data and EV_OK:
            fw.evolving = EvolvingIndexEngine.from_dict(data["evolving"])

        # NeuralMarkov 복원
        if "nm" in data and NM_OK:
            nm = data["nm"]
            fw.nm = NeuralMarkovEngine()
            fw.nm.uni   = Counter(nm["uni"])
            fw.nm.bi    = defaultdict(Counter,
                            {k:Counter(v) for k,v in nm["bi"].items()})
            fw.nm.tri   = defaultdict(Counter,
                            {k:Counter(v) for k,v in nm["tri"].items()})
            fw.nm.total = nm["total"]
            fw.nm.alpha = nm.get("alpha",0.001)
            fw.nm.mu    = nm.get("mu",  0.0)
            fw.nm.std   = nm.get("std", 1.0)
            fw.nm.is_trained = True
            if fw.nm.mu == 0.0 and fw.corpus_text:
                fw.nm._calibrate(fw.corpus_text)

        # CoreAI v2 복원
        if "coreai_v2" in data and V2_OK:
            try:
                fw.coreai_v2 = CoreAIv2Engine.load_from_dict(data["coreai_v2"])
            except Exception:
                fw.coreai_v2 = None

        # XAI + CoreAI 레이어
        fw._rebuild_layers()
        # 검색 엔진 복원
        if fw.corpus_text:
            fw.searcher = CorpusSearcher()
            fw.searcher.build(fw.corpus_text)
        return fw

    # ── 상태 요약 ─────────────────────────────────────────────
    def summary(self) -> dict:
        ev_sum = self.evolving.summary() if self.evolving else {}
        return {
            "cycle":         self.cycle,
            "nm_vocab":      len(self.nm.uni) if self.nm else 0,
            "nm_trained":    self.nm.is_trained if self.nm else False,
            **{f"ev_{k}":v for k,v in ev_sum.items()},
        }


# ── 코퍼스 검색 엔진 ─────────────────────────────────────────
class CorpusSearcher:
    """
    TF-IDF 기반 코퍼스 검색
    질문 → 가장 유사한 코퍼스 문장 반환
    환각 없음 / LLM 불필요
    """
    def __init__(self):
        self.sentences: list = []
        self.tfidf:     object = None
        self.vecs:      object = None

    def build(self, corpus_text: str):
        import re
        from collections import Counter
        import numpy as np

        # 문장 분리
        sents = list(dict.fromkeys(
            s.strip() for s in corpus_text.split('\n')
            if s.strip() and len(s.strip()) >= 6
        ))
        self.sentences = sents
        if not sents:
            return

        # 어휘 + TF-IDF
        all_toks = tokenize(corpus_text)
        cnt      = Counter(all_toks)
        vocab    = {w:i for i,w in enumerate(
            w for w,c in cnt.most_common() if c >= 1)}
        V = len(vocab)
        N = len(sents)

        # IDF
        df = Counter()
        for s in sents:
            for t in set(tokenize(s)):
                if t in vocab: df[t] += 1
        idf = {w: np.log((N+1)/(df.get(w,0)+1))+1 for w in vocab}

        # 문장 벡터
        def tv(text):
            v   = np.zeros(V)
            toks= tokenize(text); ct = Counter(toks)
            for w,c in ct.items():
                if w in vocab:
                    v[vocab[w]] = (c/max(len(toks),1))*idf.get(w,1.0)
            norm = np.linalg.norm(v)
            return v/norm if norm>1e-12 else v

        self.vocab = vocab
        self.idf   = idf
        self.vecs  = np.array([tv(s) for s in sents])

    def search(self, query: str, topk: int = 3) -> list:
        """질문과 가장 유사한 문장들 반환"""
        if not self.sentences or self.vecs is None:
            return []
        import numpy as np
        from collections import Counter

        # 쿼리 벡터
        V   = len(self.vocab)
        v   = np.zeros(V)
        toks= tokenize(query); ct = Counter(toks)
        for w,c in ct.items():
            if w in self.vocab:
                v[self.vocab[w]] = (c/max(len(toks),1))*self.idf.get(w,1.0)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return []
        v /= norm

        sims = self.vecs @ v
        idx  = np.argsort(-sims)[:topk]
        return [(self.sentences[i], float(sims[i]))
                for i in idx if sims[i] > 0.01]

    def answer(self, query: str, nm=None,
               topk: int = 5) -> dict:
        """
        검색 + NM 검증 → 최적 답변 반환
        """
        import time
        t0      = time.perf_counter()
        results = self.search(query, topk=topk)

        if not results:
            return {"answer": "관련 내용을 찾을 수 없어요.",
                    "source": "none", "score": 0.0,
                    "status": "SKIP", "ms": 0}

        # NM 검증: PASS인 것 우선
        best_ans   = results[0][0]
        best_score = results[0][1]
        best_status= "SKIP"

        if nm and nm.is_trained:
            for sent, score in results:
                r = nm.evaluate(sent)
                if r["status"] == "PASS":
                    best_ans    = sent
                    best_score  = score
                    best_status = "PASS"
                    break
                elif r["status"] == "WARNING" and best_status != "PASS":
                    best_ans    = sent
                    best_score  = score
                    best_status = "WARNING"
        else:
            best_status = "SKIP"

        ms = (time.perf_counter()-t0)*1000
        return {
            "answer":    best_ans,
            "source":    "retrieval",
            "score":     best_score,
            "status":    best_status,
            "all":       results,
            "ms":        ms,
        }
