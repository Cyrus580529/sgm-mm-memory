"""Official grading of MemEye and MobileMem-Omni, for local end-to-end runs.

Prompt texts are fetched once as plain text from the benchmarks' repositories (MemEye: benchmark/llm_judge.txt;
MobileMem: omni/eval/eval/question_answering_and_judge_prompts.txt) into a cache directory and filled in as
templates; nothing from those repositories is executed. Choice extraction follows MemEye's benchmark/evaluator.py.
"""
import json
import os
import re
import string
import subprocess
from typing import Optional


def _fetch(repo: str, path: str, dst: str) -> str:
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        text = subprocess.run(["gh", "api", f"repos/{repo}/contents/{path}", "-H", "Accept: application/vnd.github.raw"],
                              check=True, capture_output=True, text=True).stdout
        with open(dst, "w") as f:
            f.write(text)
    return open(dst).read()


class MemEyeJudge:
    """Open questions: LLM judge with scores in {0, 0.25, 0.5, 0.75, 1}. MCQ: exact match on the extracted letter."""

    def __init__(self, cache_dir: str):
        self.template = _fetch("MinghoKwok/MemEye", "benchmark/llm_judge.txt", os.path.join(cache_dir, "llm_judge.txt"))

    def prompt(self, question: str, gold: str, pred: str) -> str:
        return (self.template.replace("{{question}}", question).replace("{{ground_truth}}", gold)
                .replace("{{model_output}}", pred))

    @staticmethod
    def parse(text: str) -> float:
        m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text or "", re.S)
        try:
            v = float(json.loads(m.group(0))["score"]) if m else None
        except (ValueError, KeyError):
            v = None
        if v is None:
            m = re.search(r"score\"?\s*[:=]\s*([01](?:\.\d+)?)", text or "", re.I)
            v = float(m.group(1)) if m else 0.0
        return min(max(v, 0.0), 1.0)

    @staticmethod
    def extract_choice(text: str, valid: Optional[set] = None) -> str:
        t = (text or "").strip().upper()
        valid = valid or set(string.ascii_uppercase)
        if t in valid:
            return t
        keys = "|".join(re.escape(k) for k in sorted(valid))
        for pat in (rf"^({keys})[\.\)\:\s,]", rf"(?:ANSWER|CHOICE)\s*(?:IS|:)\s*({keys})\b", rf"\b({keys})\b"):
            m = re.search(pat, t) if not pat.startswith("^") else re.match(pat, t)
            if m:
                return m.group(1)
        return "INVALID"


class MobileMemJudge:
    """LLM judge returning CORRECT/WRONG; full coverage of the gold answer and no redundant content required."""

    def __init__(self, cache_dir: str):
        text = _fetch("zjunlp/MobileMem", "omni/eval/eval/question_answering_and_judge_prompts.txt",
                      os.path.join(cache_dir, "mobilemem_prompts.txt"))
        self.template = text[text.index("Your task is to classify"):].strip()

    def prompt(self, question: str, gold: str, evidence: str, pred: str) -> str:
        return (self.template.replace("[Question]", question).replace("[Gold Answer]", gold)
                .replace("[Evidences]", evidence).replace("[Generated Answer]", pred))

    @staticmethod
    def parse(text: str) -> int:
        m = re.search(r'"label"\s*:\s*"?(CORRECT|WRONG)', text or "", re.I)
        if m:
            return int(m.group(1).upper() == "CORRECT")
        t = (text or "").upper()
        return int("CORRECT" in t and "WRONG" not in t)
