import numpy as np
import pytest

from src.features import NUM_TOKEN, PHONE_TOKEN, URL_TOKEN, tokenize


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Hello, World! It's me.") == ["hello", "world", "it", "me"]

    def test_drops_single_character_tokens(self):
        assert tokenize("I c u r ok") == ["ok"]

    def test_phone_numbers_become_one_token(self):
        assert tokenize("Call 09061701461 now") == ["call", PHONE_TOKEN, "now"]
        assert tokenize("Call 0800 169 6031 today") == ["call", PHONE_TOKEN, "today"]

    def test_short_numbers_and_prices_become_num(self):
        # 1000 and 150 both become __num__; the lone "p" of "150p" is a 1-char token and is dropped
        assert tokenize("win £1000 or 150p") == ["win", NUM_TOKEN, "or", NUM_TOKEN]
        assert tokenize("cost 1.50 per msg, 20,000 pounds") == ["cost", NUM_TOKEN, "per", "msg", NUM_TOKEN, "pounds"]

    def test_urls_become_one_token(self):
        assert tokenize("visit www.areyouunique.co.uk now") == ["visit", URL_TOKEN, "now"]
        assert tokenize("go to http://img.sms.ac/W/jd") == ["go", "to", URL_TOKEN]

    def test_placeholder_tokens_survive_splitting(self):
        toks = tokenize("Txt CLAIM to 87066 or see www.ldew.com")
        assert NUM_TOKEN in toks and URL_TOKEN in toks

    def test_empty_and_whitespace(self):
        assert tokenize("") == []
        assert tokenize("   ") == []


from src.features import TfidfFeaturizer  # noqa: E402

CORPUS = [
    "Free entry in a weekly competition, text WIN to 80086 now",
    "Ok lar... Joking wif u oni...",
    "URGENT! You have won a free prize, call 09061701461 to claim",
    "Are we still meeting for dinner tonight?",
    "free free free call now",
    "I'll call you later tonight, ok?",
]


class TestTfidfFeaturizer:
    def test_matches_sklearn_with_same_tokenizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        ours = TfidfFeaturizer(max_features=None, min_df=2).fit(CORPUS)
        ref = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=2).fit(CORPUS)
        assert ours.feature_names_ == list(ref.get_feature_names_out())
        np.testing.assert_allclose(ours.idf_, ref.idf_, rtol=1e-6)
        np.testing.assert_allclose(ours.transform(CORPUS), ref.transform(CORPUS).toarray(), atol=1e-6)

    def test_vocabulary_comes_from_fit_texts_only(self):
        feat = TfidfFeaturizer(max_features=None, min_df=1).fit(["hello world", "hello there"])
        X = feat.transform(["hello unseen words"])
        assert X.shape == (1, 3)
        assert X[0, feat.vocabulary_["hello"]] > 0
        assert "unseen" not in feat.vocabulary_

    def test_max_features_keeps_most_frequent_terms(self):
        feat = TfidfFeaturizer(max_features=2, min_df=1).fit(CORPUS)
        assert feat.feature_names_ == ["call", "free"]      # 3 and 5 occurrences, alphabetical order

    def test_rows_are_l2_normalised_and_float32(self):
        X = TfidfFeaturizer(max_features=None, min_df=1).fit_transform(CORPUS)
        assert X.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)

    def test_all_oov_row_is_zero_not_nan(self):
        feat = TfidfFeaturizer(max_features=None, min_df=1).fit(["hello world"])
        X = feat.transform(["completely different"])
        assert not np.isnan(X).any() and X.sum() == 0

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            TfidfFeaturizer().transform(["x"])

    def test_explain_returns_nonzero_terms_largest_first(self):
        feat = TfidfFeaturizer(max_features=None, min_df=1).fit(CORPUS)
        terms = feat.explain("free free call")
        assert terms[0][0] == "free" and terms[0][1] >= terms[1][1]
        assert all(w > 0 for _, w in terms)
