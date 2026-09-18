import hashlib

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def canonicalize(document):
    return client.post("/canonicalize", json={"document": document})


def canonicalize_raw(body: bytes):
    return client.post(
        "/canonicalize", content=body, headers={"Content-Type": "application/json"}
    )


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_rfc8785_sample_document():
    r = canonicalize(
        {
            "numbers": [333333333.33333329, 1e30, 4.5, 0.002, 1e-27],
            "string": "€$\x0f\nA'B\"\\\\\"/",
            "literals": [None, True, False],
        }
    )
    assert r.status_code == 200
    expected = (
        '{"literals":[null,true,false],'
        '"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        '"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    )
    assert r.json() == {"canonical": expected, "sha256": sha256(expected)}


def test_object_keys_sorted_by_utf16_code_units():
    # UTF-16 code unit order: "z" (007A) < "é" (00E9) < "\U00010000"
    # (surrogate pair D800 DC00) < "￿" (FFFF). Code point order would
    # place "￿" before "\U00010000".
    r = canonicalize({"￿": 1, "z": 2, "\U00010000": 3, "é": 4})
    assert r.status_code == 200
    assert r.json()["canonical"] == '{"z":2,"é":4,"\U00010000":3,"￿":1}'


def test_key_order_is_deterministic_regardless_of_input_order():
    first = canonicalize({"b": 1, "a": {"d": 4, "c": 3}})
    second = canonicalize({"a": {"c": 3, "d": 4}, "b": 1})
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["canonical"] == '{"a":{"c":3,"d":4},"b":1}'


def test_equal_documents_with_different_number_syntax_match():
    integer = canonicalize_raw(b'{"document": {"n": 1, "x": 1e2}}')
    decimal = canonicalize_raw(b'{"document": {"x": 100.0, "n": 1.0}}')
    assert integer.status_code == decimal.status_code == 200
    assert integer.json() == decimal.json()
    assert integer.json()["canonical"] == '{"n":1,"x":100}'


def test_numbers_use_ecmascript_canonical_form():
    r = canonicalize(
        [1.0, -0.0, 100, 1e20, 1e21, 0.000001, 1e-7, -4.5, 77777777777777777777777777777]
    )
    assert r.status_code == 200
    assert r.json()["canonical"] == (
        "[1,0,100,100000000000000000000,1e+21,0.000001,1e-7,-4.5,"
        "7.777777777777778e+28]"
    )


def test_strings_use_only_mandatory_escapes():
    r = canonicalize('"\\\b\t\n\f\r\x00\x1f\x7fé€😀/')
    assert r.status_code == 200
    assert r.json()["canonical"] == (
        '"\\"\\\\\\b\\t\\n\\f\\r\\u0000\\u001f\x7fé€😀/"'
    )


def test_array_order_and_literals_preserved():
    r = canonicalize([3, [2, 1], True, False, None])
    assert r.status_code == 200
    assert r.json()["canonical"] == "[3,[2,1],true,false,null]"


def test_bare_scalar_document():
    assert canonicalize("hello").json()["canonical"] == '"hello"'
    assert canonicalize(None).json()["canonical"] == "null"
    assert canonicalize(True).json()["canonical"] == "true"


def test_empty_containers_and_nesting():
    r = canonicalize({"a": {}, "b": [], "c": [{}]})
    assert r.status_code == 200
    assert r.json()["canonical"] == '{"a":{},"b":[],"c":[{}]}'


def test_sha256_matches_utf8_bytes_of_canonical_text():
    r = canonicalize({"é": "✓", "n": 1e30})
    assert r.status_code == 200
    body = r.json()
    assert body["sha256"] == sha256(body["canonical"])


def test_missing_document_field_is_422():
    r = client.post("/canonicalize", json={})
    assert r.status_code == 422


def test_malformed_json_body_is_422():
    r = canonicalize_raw(b'{"document": ')
    assert r.status_code == 422


def test_non_object_body_is_422():
    r = client.post("/canonicalize", json=[1, 2, 3])
    assert r.status_code == 422


def test_lone_surrogate_is_422():
    r = canonicalize_raw(b'{"document": "\\ud800"}')
    assert r.status_code == 422
    assert r.json()["detail"] == "document cannot be canonicalized"


def test_lone_surrogate_in_object_key_is_422():
    r = canonicalize_raw(b'{"document": {"\\ud800": 1}}')
    assert r.status_code == 422


def test_non_finite_number_is_422():
    r = canonicalize_raw(b'{"document": 1e999}')
    assert r.status_code == 422


def test_integer_beyond_double_range_is_422():
    r = canonicalize(10**400)
    assert r.status_code == 422
