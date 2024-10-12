import torch
import torch.nn as nn


class CRF(nn.Module):

    def __init__(self, num_tags: int, batch_first: bool = False):
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.randn(self.num_tags))
        self.end_transitions = nn.Parameter(torch.randn(self.num_tags))
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags))

    def _forward_alg(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _score_sentence(self, emissions: torch.Tensor, tags: torch.LongTensor,
                        mask: torch.ByteTensor) -> torch.Tensor:

        seq_length, batch_size = tags.shape

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor):
        mask = mask.type(torch.bool)
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]

        history: torch.Tensor = None
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history = indices.unsqueeze(0) if history is None \
                else torch.cat((history, indices.unsqueeze(0)), 0)

        score += self.end_transitions
        return score, history

    def log_likelihood(self, emissions: torch.tensor, tags: torch.LongTensor, mask: torch.ByteTensor = None,
                       reduction: str = 'sum') -> torch.Tensor:

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_alg(emissions, mask)
        llh = gold_score - forward_score

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def forward(self, emissions: torch.FloatTensor, mask: torch.ByteTensor = None):
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        score, history = self._viterbi_decode(emissions, mask)
        seq_length = emissions.size(0)
        _, best_last_tag = score.max(dim=1)
        best_tags = best_last_tag.unsqueeze(-1)
        for i in reversed(range(0, seq_length - 1)):
            best_last_tag = torch.gather(history[i], dim=1, index=best_tags[:, -1].unsqueeze(-1))
            best_tags = torch.cat((best_tags, best_last_tag), 1)
        best_tags_list = torch.flip(best_tags, dims=[1])

        return best_tags_list


class FCRF(CRF):

    def __init__(self, num_tags: int, batch_first: bool = False, sub2intents: torch.LongTensor = None):
        if sub2intents is None:
            raise ValueError(f'sub2intents can not be None')
        super(FCRF, self).__init__(num_tags, batch_first)
        self.sub2intents = sub2intents

    def _forward_cls(self, emissions: torch.Tensor) -> torch.Tensor:

        emissions = emissions.transpose(0, 1)
        true_labels = self.sub2intents.transpose(0, 1).to(emissions.device)
        batch_size, seq_length, _ = emissions.size()

        score = self.start_transitions[true_labels[0]] + \
                torch.index_select(emissions[:, 0], 1, true_labels[0])
        for i in range(1, seq_length):
            score += self.transitions[true_labels[i - 1], true_labels[i]]
            score += torch.index_select(emissions[:, i], 1, true_labels[i])
        score += self.end_transitions[true_labels[seq_length - 1]]

        return score

    def log_likelihood(self, emissions: torch.tensor, tags: torch.LongTensor, mask: torch.ByteTensor = None,
                       reduction: str = 'sum') -> torch.Tensor:

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        gold_score = self._score_sentence(emissions, tags, mask)
        labels_scores = self._forward_cls(emissions)
        forward_score, _ = labels_scores.max(dim=1)
        llh = gold_score - forward_score

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def forward(self, emissions: torch.FloatTensor, mask: torch.ByteTensor = None):

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
        pred_scores = self._forward_cls(emissions)

        return pred_scores
