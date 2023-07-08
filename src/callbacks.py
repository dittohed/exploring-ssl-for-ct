import torch


class EarlyStopping():
    def __init__(self, patience):
        self._patience = patience
        self._best_score = 0
        self._counter = 0
        self.terminate = False

    def __call__(self, score):
        if score <= self._best_score:
            self._counter += 1
            if self._counter == self._patience:
                print(f'Score has not increased from {self._best_score} for {self._patience} eval steps.')
                self.terminate = True
        else:
            self._best_score = score
            self._counter = 0
    

class BestCheckpoint():
    def __init__(self, model, save_path):
        self._model = model
        self._save_path = save_path
        self._best_score = 0

        self._save_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, score):
        if score > self._best_score:
            print(f'Score has increased from {self._best_score} to {score}.')
            self._best_score = score
            torch.save(self._model.state_dict(), self._save_path)
            print(f'Successfully saved best weights under {self._save_path}.')

