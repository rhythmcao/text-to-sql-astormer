import torch


class Hypothesis():

    __slots__ = ('action', 'score')

    def __init__(self, action: list, score: torch.FloatTensor) -> None:
        self.action = action
        self.score = score.item() if torch.is_tensor(score) else score


class SEQBeam(object):
    """ Class for managing the internals of the beam search process.
        Takes care of beams, back pointers, and scores.
        @args:
            tranx (dict): obtain indices of padding, beginning, and ending.
            database (dict): number of tables and columns for the current sample.
            beam_size (int): beam size
            device (torch.device)
    """
    def __init__(self, tranx, database, beam_size=5, n_best=5, device=None, top_k=0):
        super(SEQBeam, self).__init__()
        self.tokenizer = tranx.tokenizer
        self.shifts = (
            self.tokenizer.vocab_size, 
            self.tokenizer.vocab_size + database['table'],
            self.tokenizer.vocab_size + database['table'] + database['column']
        )
        self.beam_size, self.n_best = beam_size, n_best
        self.device = device
        # The score for each translation on the beam.
        self.scores = torch.zeros(self.beam_size, dtype=torch.float, device=self.device)

        # Has EOS topped the beam yet.
        self._eos = self.tokenizer.sep_token_id
        self.eos_top = False

        # Other special symbols
        self._bos = self.tokenizer.cls_token_id
        self._pad = self.tokenizer.pad_token_id

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.zeros(self.beam_size, dtype=torch.long, device=self.device).fill_(self._pad)]
        self.next_ys[0][0] = self._bos

        # Store finished hypothesis
        self.completed_hyps = []
        self.top_k = int(top_k) if top_k >= 2 and top_k <= self.beam_size else self.beam_size


    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]


    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]


    def advance(self, word_probs):
        """ Given prob over words for every last beam `wordLk`
        Parameters:
            `word_probs`- probs of advancing from the last step (K x words)
        Returns: True if beam search is complete.
        """
        word_probs[:, self._bos], word_probs[:, self._pad] = -1e20, -1e20
        cur_top_k = self.beam_size if len(self.prev_ks) == 0 else self.top_k
        top_k, sort_key = word_probs.topk(cur_top_k, -1, True, True)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = top_k + self.scores.unsqueeze(1)
        else:
            beam_scores = top_k[0]
        flat_beam_scores = beam_scores.contiguous().view(-1)
        _, best_scores_id = flat_beam_scores.topk(self.beam_size, 0, True, True)

        # best_scores_id is flattened beam_size x cur_top_k array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id // cur_top_k
        self.prev_ks.append(prev_k)
        next_y = torch.take(sort_key.contiguous().view(-1), best_scores_id)
        self.next_ys.append(next_y)
        self.scores = torch.take(beam_scores.contiguous().view(-1), best_scores_id)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i].item() == self._eos:
                self.completed_hyps.append((self.scores[i], len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS
        if self.next_ys[-1][0].item() == self._eos:
            self.eos_top = True

        return self.done


    @property
    def done(self):
        return self.eos_top and len(self.completed_hyps) >= self.n_best


    def sort_finished(self):
        if len(self.completed_hyps) > 0:
            self.completed_hyps.sort(key=lambda a: - a[0]) # / len(a[1]))
            completed_hyps = [Hypothesis(action=self.get_hyp(t, k), score=s) for s, t, k in self.completed_hyps]
        else:
            completed_hyps = [Hypothesis(action=[self._bos, self._eos], score=-1e10)]
        return completed_hyps


    def get_hyp(self, timestep, k):
        """ Walk back to construct the full hypothesis. 
            hyp contains [SEP] but does not contain [CLS]
            @return:
                hyp: list of id
        """
        hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k].item())
            k = self.prev_ks[j][k]
        return hyp[::-1]