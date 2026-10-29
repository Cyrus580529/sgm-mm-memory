"""Date expressions in questions -> periods used as a retrieval signal."""
from datetime import date

import pytest

from app.retrieval import query_periods, strip_anchor


def hit(q, d):
    return any(p(d) for p in query_periods(q))


@pytest.mark.parametrize("q,inside,outside", [
    ("Which of the pictures shared in July 2024?", date(2024, 7, 3), date(2023, 7, 3)),
    ("in early August 2024", date(2024, 8, 5), date(2024, 8, 15)),
    ("What happened on 2024/05/07?", date(2024, 5, 7), date(2024, 5, 8)),
    ("Today is May 2, 2024, I had a formal last week", date(2024, 4, 23), date(2024, 4, 10)),
    ("Today is 2024/05/31 (Fri) 07:58. What did I buy last week?", date(2024, 5, 22), date(2024, 5, 10)),
    ("Today is Jan 5, 2024. What did I buy in December?", date(2023, 12, 9), date(2024, 12, 9)),
    ("Today is 2023-03-10. What did I eat 3 days ago?", date(2023, 3, 7), date(2023, 3, 1)),
    ("我7月2号晚上和蒋立新聊了什么？", date(2025, 7, 2), date(2025, 7, 3)),
    ("2025年5月20日晚上我写了什么？", date(2025, 5, 20), date(2024, 5, 20)),
    ("今天是2025年8月10日。我上周去了哪家餐厅？", date(2025, 8, 4), date(2025, 7, 20)),
    ("今天是2025年1月5日。我12月底买了什么？", date(2024, 12, 28), date(2025, 12, 28)),
    ("今天是2025年3月1日。三天前我做了什么？", date(2025, 2, 26), date(2025, 2, 10)),
])
def test_periods(q, inside, outside):
    assert hit(q, inside) and not hit(q, outside)


def test_no_period():
    assert query_periods("What may I do about my dog?") == []
    assert query_periods("我那次给父母转钱，主要想补贴哪些日常开销？") == []


def test_strip_anchor():
    assert strip_anchor("Today is 2024/05/31 (Fri) 07:58. How much did I spend?") == "How much did I spend?"
    assert strip_anchor("今天是2025年8月10日。我上周去了哪家餐厅？") == "我上周去了哪家餐厅？"
    assert strip_anchor("What did I buy?") == "What did I buy?"
