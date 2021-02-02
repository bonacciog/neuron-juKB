#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Musixmatch AI API
#
# Copyright (c) 2019 Musixmatch spa
#


import torch
import torch.neuron

class NeuronLSTM(torch.nn.Module):

    def __init__(
            self,
            num_layers=3,
            num_embeddings=32,
            bidirectional=True,
            hidden_size=128,
            padding_idx=0,
            padding_value=0,
            embed_dim=32,
            left_pad=1):
        super().__init__()

        #self.src_lengths = src_lengths

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = torch.nn.Embedding(
            num_embeddings, embed_dim, padding_idx=self.padding_idx)

        self.lstm = torch.nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        # if self.left_pad:
        # convert left-padding to right-padding
        # src_tokens = convert_padding_direction(
        #    src_tokens,
        #    self.padding_idx,
        #    left_to_right=True,
        # )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(
            packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat([
                    torch.cat([outs[2 * i], outs[2 * i + 1]],
                              dim=0).view(1, bsz, self.output_units)
                    for i in range(self.num_layers)
                ], dim=0)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        output = [sentemb, x, final_hiddens, final_cells]

        if encoder_padding_mask.any():
            output.append(encoder_padding_mask)

        # return {
        #    'sentemb': sentemb,
        #    'encoder_out': (x, final_hiddens, final_cells),
        #    'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        # }

        return output