from datetime import datetime, timedelta
import numpy as np


class Accounts:
    def __init__(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2024-02-01",
        frequency: str = "daily",
    ):
        # self.document_definition = document_definition
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.frequency = frequency
        self.current_date = self.start_date
        self.documents = {
            "checkingAccount": {
                "data": {
                    "date": self.start_date,
                    "accountBalance": 200,
                    "transferToSavings": 30,
                    "expenses": -103,
                    "creditCardPayments": 10,
                }
            },
            "creditCardAccount": {
                "data": {"date": self.start_date, accountBalance": 100, "payments": -10, "expenses": 100},
            },
            "savingsAccount": {
                "data": {"date": self.start_date, "accountBalance": 3401, "transferFromChecking": 30}
            },
        }

    @property
    def finished(self):
        return self.current_date >= self.end_date

    def increment_date(self, increment=1):
        # if self.frequency == "daily":
        self.current_date += timedelta(days=increment)

    def synthesize(self):
        while not self.finished:
            self.increment_date
            print(self.current_date)
            checking = self.documents["checkingAccount"]["data"].copy()
            savings = self.documents["savingsAccount"]["data"].copy()
            credit = self.documents["creditCardAccount"]["data"].copy()

            checking["date"] = self.current_date
            savings["date"] = self.current_date
            credit["date"] = self.current_date

            if self.current_date.weekday() == 4:  # get paid on fridays
                today_checking["accountBalance"] += 400
                savings_transfer = 100 + np.random.normal(1, 1)
                card_payments = 50 + np.random.normal(3, 2)
            else:
                savings_transfer = 5 * np.random.normal(1, 0.1)
                card_payments = 10 + np.random.normal(1, 0.1)
            self.documents = {
                "checkingAccount": {
                    "data": {
                        "date": self.start_date,
                        "accountBalance": 200,
                        "transferToSavings": 30,
                        "expenses": -103,
                        "creditCardPayments": 10,
                    }
                },
                "creditCardAccount": {
                    "data": {"date": self.start_date, accountBalance": 100, "payments": -10, "expenses": 100},
                },
                "savingsAccount": {
                    "data": {"date": self.start_date, "accountBalance": 3401, "transferFromChecking": 30}
                },
            }

            checking_today = {
                "date": self.current_date,
                "expenses": np.random.uniform(1, 10),
                "creditCardPayments": card_payments
                "accountBalance":
            }
            # self.documents = {
            #     "checkingAccount": {
            #         "data": {"30, "expenses": -103, "creditCardPayments": 10}
            #     },
            #     "creditCardAccount": {
            #         "data": {"accountBalance": 100, "payments": -10, "expenses": 100},
            #     },
            #     "savingsAccount": {
            #
            #         "data": {"accountBalance": 3401, "transferFromChecking": 30}
            #     }
            # }

            self.increment_date()


if __name__ == "__main__":
    accounts = Accounts(start_date="2024-01-01", end_date="2024-01-10")
    accounts.synthesize()
