import csv
import json

from app.services.oasis_profile_generator import (
    OasisAgentProfile,
    OasisProfileGenerator,
    _coerce_to_str,
    _coerce_to_str_list,
)


# Non-ASCII (CJK) values are written as escapes so the source stays ASCII:
#   \u4e2d\u6587 "Chinese", \u8be6\u7ec6 "detailed", \u4eba\u8bbe "persona",
#   \u4e2d\u56fd "China", \u7814\u7a76\u5458 "researcher", \u653f\u7b56 "policy",
#   \u793e\u4f1a "society", \u516c\u5f00\u7b80\u4ecb "public bio".


def test_text_coercion_handles_none_nested_objects_and_lists():
    assert _coerce_to_str(None) == ""
    assert _coerce_to_str({"text": {"value": "\u4e2d\u6587"}}) == "\u4e2d\u6587"
    assert _coerce_to_str([None, {"description": "alpha"}, ["beta"]]) == "alpha, beta"
    assert _coerce_to_str({"unexpected": None}) == '{"unexpected": null}'


def test_list_coercion_flattens_nested_values_and_drops_missing_items():
    assert _coerce_to_str_list(
        ["AI", ["policy", None], {"name": "society"}, 4]
    ) == ["AI", "policy", "society", "4"]
    assert _coerce_to_str_list(None) == []


def test_profile_construction_is_the_single_normalization_boundary():
    profile = OasisAgentProfile(
        user_id=1,
        user_name="agent",
        name="Agent Name",
        bio=None,
        persona={"text": {"value": "\u8be6\u7ec6\u4eba\u8bbe"}},
        gender=["female"],
        mbti={"value": "INTJ"},
        country={"name": "\u4e2d\u56fd"},
        profession={"description": "\u7814\u7a76\u5458"},
        interested_topics=["AI", ["\u653f\u7b56", None], {"name": "\u793e\u4f1a"}],
    )

    assert profile.bio == "Agent Name"
    assert profile.persona == "\u8be6\u7ec6\u4eba\u8bbe"
    assert profile.gender == "female"
    assert profile.mbti == "INTJ"
    assert profile.country == "\u4e2d\u56fd"
    assert profile.profession == "\u7814\u7a76\u5458"
    assert profile.interested_topics == ["AI", "\u653f\u7b56", "\u793e\u4f1a"]
    assert "None" not in json.dumps(profile.to_dict(), ensure_ascii=False)


def test_normalized_profile_serializes_to_twitter_and_reddit(tmp_path):
    profile = OasisAgentProfile(
        user_id=1,
        user_name="agent",
        name="Agent Name",
        bio={"summary": "\u516c\u5f00\u7b80\u4ecb"},
        persona=["\u8be6\u7ec6", "\u4eba\u8bbe"],
        mbti={"text": "ENFP"},
        interested_topics=["AI", ["\u653f\u7b56"]],
    )
    generator = object.__new__(OasisProfileGenerator)
    twitter_path = tmp_path / "twitter_profiles.csv"
    reddit_path = tmp_path / "reddit_profiles.json"

    generator._save_twitter_csv([profile], str(twitter_path))
    generator._save_reddit_json([profile], str(reddit_path))

    with twitter_path.open(encoding="utf-8") as handle:
        twitter = next(csv.DictReader(handle))
    reddit = json.loads(reddit_path.read_text(encoding="utf-8"))[0]

    assert twitter["description"] == "\u516c\u5f00\u7b80\u4ecb"
    assert twitter["user_char"] == "\u516c\u5f00\u7b80\u4ecb \u8be6\u7ec6, \u4eba\u8bbe"
    assert reddit["bio"] == "\u516c\u5f00\u7b80\u4ecb"
    assert reddit["persona"] == "\u8be6\u7ec6, \u4eba\u8bbe"
    assert reddit["mbti"] == "ENFP"
    assert reddit["interested_topics"] == ["AI", "\u653f\u7b56"]
