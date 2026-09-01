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
