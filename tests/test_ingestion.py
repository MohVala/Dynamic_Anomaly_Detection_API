from utils.Ingestion.api_flatten import flatten_json


def test_flatten_simple_dict():
    data = {"a": 1, "b": {"c": 2}}
    result = flatten_json(data)

    assert result == {
        "a": 1,
        "b_c": 2
    }
