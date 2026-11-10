import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import auth
from dashboard.server import app
import dashboard.server as server
from werkzeug.security import generate_password_hash


class FakeUsers:
    def __init__(self):
        self.documents = []

    def create_index(self, *_args, **_kwargs):
        return None

    def find_one(self, query):
        for document in self.documents:
            if self._matches(document, query):
                return document
        return None

    def insert_one(self, document):
        self.documents.append(document)

    def update_one(self, query, update):
        document = self.find_one(query)
        if not document:
            return
        for key, value in update.get("$set", {}).items():
            document[key] = value
        for key in update.get("$unset", {}):
            document.pop(key, None)
        for key, value in update.get("$inc", {}).items():
            document[key] = document.get(key, 0) + value
        for key, value in update.get("$addToSet", {}).items():
            document.setdefault(key, [])
            if value not in document[key]:
                document[key].append(value)
        for key, value in update.get("$pull", {}).items():
            document[key] = [item for item in document.get(key, []) if item != value]

    @classmethod
    def _matches(cls, document, query):
        for key, value in query.items():
            if key == "$or":
                if not any(cls._matches(document, choice) for choice in value):
                    return False
            elif isinstance(value, dict) and "$gte" in value:
                if document.get(key, 0) < value["$gte"]:
                    return False
            elif document.get(key) != value:
                return False
        return True


class AuthTests(unittest.TestCase):
    def test_account_verification_login_and_per_user_watchlists(self):
        users = FakeUsers()
        users.documents.append({
            "_id": "second-user",
            "username": "second",
            "username_key": "second",
            "email": "second@example.com",
            "email_key": "second@example.com",
            "password_hash": generate_password_hash("another-good-password"),
            "verified": True,
            "watchlist": [],
        })
        sent_codes = []

        with patch.dict(app.config, {"SECRET_KEY": "test-secret"}), \
             patch.object(auth, "get_users", return_value=users), \
             patch.object(auth, "_send_verification_email",
                          side_effect=lambda _email, _username, code: sent_codes.append(code)), \
             patch.object(server, "get_user",
                          side_effect=lambda user_id: users.find_one({
                              "_id": user_id, "verified": True
                          })):
            first_client = app.test_client()
            signup = first_client.post("/api/auth/signup", json={
                "username": "first_user",
                "email": "first@example.com",
                "password": "a-good-password",
            })
            self.assertEqual(signup.status_code, 201)
            self.assertEqual(len(sent_codes), 1)

            unverified_login = first_client.post("/api/auth/login", json={
                "identifier": "first_user",
                "password": "a-good-password",
            })
            self.assertEqual(unverified_login.status_code, 403)

            verified = first_client.post("/api/auth/verify", json={
                "email": "first@example.com",
                "code": sent_codes[0],
            })
            self.assertEqual(verified.status_code, 200)

            login = first_client.post("/api/auth/login", json={
                "identifier": "first@example.com",
                "password": "a-good-password",
            })
            self.assertEqual(login.status_code, 200)
            csrf_token = login.get_json()["csrfToken"]
            added = first_client.post(
                "/api/watchlist",
                json={"ticker": "aapl"},
                headers={"X-CSRF-Token": csrf_token},
            )
            self.assertEqual(added.get_json()["tickers"], ["AAPL"])

            second_client = app.test_client()
            second_login = second_client.post("/api/auth/login", json={
                "identifier": "second",
                "password": "another-good-password",
            })
            self.assertEqual(second_login.status_code, 200)
            second_watchlist = second_client.get("/api/watchlist")
            self.assertEqual(second_watchlist.get_json()["tickers"], [])
            first_user = users.find_one({"email_key": "first@example.com"})
            self.assertEqual(first_user["watchlist"], ["AAPL"])

    def test_api_requires_login_and_csrf_for_changes(self):
        client = app.test_client()
        self.assertEqual(client.get("/api/tickers").status_code, 401)

        with patch.dict(app.config, {"SECRET_KEY": "test-secret"}), \
             patch.object(server, "get_user", return_value={"_id": "user", "verified": True}):
            with client.session_transaction() as browser_session:
                browser_session["user_id"] = "user"
                browser_session["csrf_token"] = "expected-token"
            response = client.post("/api/watchlist", json={"ticker": "AAPL"})

        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
