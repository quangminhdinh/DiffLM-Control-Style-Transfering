from .trainer import trainer
from .config import test_set


if __name__ == "__main__":
    result = trainer.predict(test_set)
    print(result)
