"""MemLens official LLM-judge prompts, used to score local end-to-end runs the way the benchmark does.

The prompt texts live in the MemLens repository (github.com/xrenaf/MEMLENS, MIT, judge_prompts/*.py). They are
fetched once as plain text into the data directory and read with the Python parser (string literals only; nothing
from those files is imported or executed). The task-key mapping and case formatting follow judge_prompts/__init__.py.
"""
import ast
import os
import re
import subprocess

FILES = ["base.py", "information_extraction.py", "knowledge_update.py", "multi_session_reasoning.py",
         "temporal_reasoning.py", "answer_refusal.py"]


def _fetch(dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for f in FILES:
        path = os.path.join(dst_dir, f)
        if not os.path.exists(path):
            text = subprocess.run(["gh", "api", f"repos/xrenaf/MEMLENS/contents/judge_prompts/{f}",
                                   "-H", "Accept: application/vnd.github.raw"],
                                  check=True, capture_output=True, text=True).stdout
            with open(path, "w") as fh:
                fh.write(text)


def _literals(path: str) -> dict:
    """Module-level NAME = <literal> assignments, evaluated as literals only."""
    out = {}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


class Judge:
    def __init__(self, cache_dir: str):
        _fetch(cache_dir)
        self.base = _literals(os.path.join(cache_dir, "base.py"))["QA_BENCHMARK_JUDGE_PROMPT"]
        self.criteria, self.examples = {}, {}
        for f in FILES[1:]:
            lit = _literals(os.path.join(cache_dir, f))
            self.criteria.update(lit.get("CRITERIA", {}))
            self.examples.update(lit.get("EXAMPLES", {}))

    @staticmethod
    def task_key(qtype: str, subtype: str, reference: str) -> str:
        if qtype == "multi_session_reasoning":
            if subtype == "arithmetic":
                return "MSR_Arithmetic"
            if subtype == "entity_resolution" and reference.strip() in ("Yes", "No"):
                return "MSR_YesNo"
            return "MSR_Counting"
        if qtype == "temporal_reasoning":
            return {"duration_comparison": "TR_DurationComparison",
                    "order_ranking": "TR_OrderRanking"}.get(subtype, "TR_DateExtraction")
        if qtype == "knowledge_update":
            return "KU_KnowledgeUpdate"
        if qtype == "answer_refusal":
            return "AR_AnswerRefusal"
        return "IE_PreviousInfo" if subtype == "previnfo" else "IE_Entity"

    def prompt(self, key: str, question: str, reference: str, prediction: str, old_answer: str = "") -> str:
        system = self.base.format(task_criteria=self.criteria.get(key, ""), examples=self.examples.get(key, ""))
        if key == "KU_KnowledgeUpdate" and old_answer:
            case = (f"[Current Case]\n<Question>: {question}\n<Standard Answer>: {reference}\n"
                    f"<Old (Outdated) Answer>: {old_answer}\n<Student Answer>: {prediction}\n\n")
        elif key == "AR_AnswerRefusal":
            case = (f"[Current Case]\n<Question>: {question}\n<Standard Answer>: Insufficient information\n"
                    f"<Student Answer>: {prediction}\n\n")
        else:
            case = (f"[Current Case]\n<Question>: {question}\n<Standard Answer>: {reference}\n"
                    f"<Student Answer>: {prediction}\n\n")
        return system + case + "[Scoring Rationale]:"

    @staticmethod
    def parse(text: str) -> int:
        for pat in (r'"answer_score"\s*:\s*(\d+)', r"deserves\s+(\d+)\s+point", r"\[Score\]\s*:\s*(\d+)"):
            m = re.search(pat, text or "", re.I)
            if m:
                return min(int(m.group(1)), 1)
        return 0
