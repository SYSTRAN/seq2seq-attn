#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the data for the LSTM.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
import math
import re
from collections import defaultdict

class Indexer:
    def __init__(self, symbols = ["<blank>","<unk>","<s>","</s>"]):
        self.vocab = defaultdict(int)
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOS = symbols[2]
        self.EOS = symbols[3]
        self.d = {self.PAD: 1, self.UNK: 2, self.BOS: 3, self.EOS: 4}
        self.max_num_values = 0

    def add_w(self, ws):
        for w in ws:
            if w not in self.d:
                self.d[w] = len(self.d) + 1

    def convert(self, w):
        return self.d[w] if w in self.d else self.d[self.UNK]

    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]

    def clean(self, s):
        s = s.replace(self.PAD, "")
        s = s.replace(self.BOS, "")
        s = s.replace(self.EOS, "")
        return s

    def write(self, outfile, chars=0):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k.encode('utf-8'), v #always keep utf8 encode/decode
        out.close()

    def prune_vocab(self, k):
        vocab_list = [(word, count) for word, count in self.vocab.iteritems()]
        vocab_list.sort(key = lambda x: x[1], reverse=True)
        k = min(k, len(vocab_list))
        self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list[:k]}
        for word in self.pruned_vocab:
            if word not in self.d:
                self.d[word] = len(self.d) + 1

    def load_vocab(self, vocab_file, chars=0):
        self.d = {}
        for line in open(vocab_file, 'r'):
            v, k = line.decode("utf-8").strip().split() #always keep utf8 encode/decode
            self.d[v] = int(k)

def pad(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]
    return ls + [symbol] * (length -len(ls))

def save_features(name, indexers, outputfile):
    if len(indexers) > 0:
        print("Number of additional features on {} side: {}".format(name, len(indexers)))
    for i in range(len(indexers)):
        indexers[i].write(outputfile + "." + name + "_feature_" + str(i+1) + ".dict", )
        print(" * {} feature {} of size: {} (maximal number of values per word: {})".format(name, i+1, len(indexers[i].d), indexers[i].max_num_values))

def load_features(name, indexers, outputfile):
    for i in range(len(indexers)):
        indexers[i].load_vocab(outputfile + "." + name + "_feature_" + str(i+1) + ".dict", )
        print(" * {} feature {} of size: {} (maximal number of values per word: {})".format(name, i+1, len(indexers[i].d), indexers[i].max_num_values))

def get_data(args):
    int_max_size = 31
    src_feature_indexers = []
    src_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])
    target_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])
    target_feature_indexers = []
    char_indexer = Indexer(["<blank>","<unk>","{","}"])
    char_indexer.add_w([src_indexer.PAD, src_indexer.UNK, src_indexer.BOS, src_indexer.EOS])

    def init_feature_indexers(indexers, count):
        for i in range(count):
            indexers.append(Indexer(["<blank>","<unk>","<s>","</s>"]))

    def load_sentence(sent, indexers):
        sent_seq = sent.strip().split()
        sent_words = ''
        sent_features = []

        for entry in sent_seq:
            fields = entry.split('-|-')
            word = fields[0]
            sent_words += (' ' if sent_words else '') + word

            if len(fields) > 1:
                count = len(fields) - 1
                if len(sent_features) == 0:
                    sent_features = [ [] for i in range(count) ]
                if len(indexers) == 0:
                    init_feature_indexers(indexers, count)

                for i in range(1, len(fields)):
                    values = fields[i].split(',')
                    indexers[i-1].max_num_values = max(len(values), indexers[i-1].max_num_values)
                    sent_features[i-1].append(values)

        return sent_words, sent_features

    def add_features_vocab(orig_features, indexers):
        if len(indexers) > 0:
            index = 0
            for features in orig_features:
                for values in features:
                    indexers[index].add_w(values)
                index += 1

    def make_vocab(srcfile, targetfile, seqlength, max_word_l=0, chars=0, train=1):
        num_sents = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig, src_orig_features = load_sentence(src_orig, src_feature_indexers)
            targ_orig, targ_orig_features = load_sentence(targ_orig, target_feature_indexers)
            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip()) #always keep utf8 encode/decode
            targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip()) #always keep utf8 encode/decode
            targ = targ_orig.strip().split()
            src = src_orig.strip().split()
            if len(targ) > seqlength or len(src) > seqlength or len(targ) < 1 or len(src) < 1:
                continue
            num_sents += 1
            if train == 1:
                for word in targ:
                    if chars == 1:
                        word = char_indexer.clean(word)
                        if len(word) == 0:
                            continue
                        max_word_l = max(len(word)+2, max_word_l)
                        for char in list(word):
                            char_indexer.vocab[char] += 1
                    target_indexer.vocab[word] += 1

                add_features_vocab(src_orig_features, src_feature_indexers)
                add_features_vocab(targ_orig_features, target_feature_indexers)

                for word in src:
                    if chars == 1:
                        word = char_indexer.clean(word)
                        if len(word) == 0:
                            continue
                        max_word_l = max(len(word)+2, max_word_l)
                        for char in list(word):
                            char_indexer.vocab[char] += 1
                    src_indexer.vocab[word] += 1

        return max_word_l, num_sents

    def convert(srcfile, targetfile, alignfile, alignpattern, storealign, batchsize, seqlength, outfile, num_sents,
                max_word_l, max_sent_l=0,chars=0, unkfilter=0, shuffle=0, verbose=0):

        def init_features_tensor(indexers):
            return [ np.zeros((num_sents,
                               newseqlength,
                               int(math.ceil(len(indexers[i].d)/float(int_max_size)))), dtype=int)
                     for i in range(len(indexers)) ]

        def values_to_identifier(values, vocab_size):
            binary = [ 1 if k+1 in values else 0 for k in range(vocab_size) ]
            splitted = [ binary[i:i+int_max_size] if i+int_max_size < vocab_size else binary[i:]
                         for i in range(0, vocab_size, int_max_size) ]
            values = []
            for k in range(len(splitted)):
                value = 0
                for i in range(len(splitted[k])):
                    value += splitted[k][i] * (2**i)
                values.append(value)
            return values

        def load_features(orig_features, indexers, seqlength):
            if len(orig_features) == 0:
                return None

            features = []
            for i in range(len(orig_features)):
                features.append([[indexers[i].UNK]] + [[indexers[i].BOS]]
                                + orig_features[i] + [[indexers[i].EOS]])

            for i in range(len(features)):
                features[i] = pad(features[i], seqlength+1, [indexers[i].PAD])
                for j in range(len(features[i])):
                    features[i][j] = indexers[i].convert_sequence(features[i][j])
                    identifier = values_to_identifier(features[i][j], len(indexers[i].d))
                    features[i][j] = identifier
                features[i] = np.array(features[i], dtype=int)
            return features

        newseqlength = seqlength + 2 #add 2 for EOS and BOS

        alignfile_hdl = None
        alignments = None
        if not alignfile == '':
            alignfile_hdl = open(alignfile,'r')
            if storealign == 1:
                alignments = np.zeros((num_sents,newseqlength,newseqlength), dtype=np.uint8)
        alignpattern_re = re.compile(alignpattern)

        targets = np.zeros((num_sents, newseqlength), dtype=int)
        targets_features = init_features_tensor(target_feature_indexers)
        targets_features_output = init_features_tensor(target_feature_indexers)
        target_output = np.zeros((num_sents, newseqlength), dtype=int)
        sources = np.zeros((num_sents, newseqlength), dtype=int)
        sources_features = init_features_tensor(src_feature_indexers)
        source_lengths = np.zeros((num_sents,), dtype=int)
        target_lengths = np.zeros((num_sents,), dtype=int)
        if chars==1:
            sources_char = np.zeros((num_sents, newseqlength, max_word_l), dtype=int)
            targets_char = np.zeros((num_sents, newseqlength, max_word_l), dtype=int)
        dropped = 0
        dropped_align = 0
        dropped_length = 0
        dropped_unk = 0
        sent_id = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig, src_orig_features = load_sentence(src_orig, src_feature_indexers)
            targ_orig, targ_orig_features = load_sentence(targ_orig, target_feature_indexers)

            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip()) #always keep utf8 encode/decode
            targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip()) #always keep utf8 encode/decode

            targw = [target_indexer.BOS] + targ_orig.strip().split() + [target_indexer.EOS]
            srcw =  [src_indexer.BOS] + src_orig.strip().split() + [src_indexer.EOS]
            max_sent_l = max(len(targw), len(srcw), max_sent_l)
            if len(targw) > newseqlength or len(srcw) > newseqlength or len(targw) < 3 or len(srcw) < 3:
                dropped += 1
                dropped_length += 1
                if verbose: print "DROP LEN\t"+src_orig.encode("utf-8").strip()+"\n"+targ_orig.encode("utf-8").strip()+"\t" #always keep utf8 encode/decode


                # skip align file
                if alignfile_hdl: alignfile_hdl.readline()
                continue

            targw = pad(targw, newseqlength+1, target_indexer.PAD)
            targ_char = []
            for word in targw:
                if chars == 1:
                    word = char_indexer.clean(word)
                #use UNK for target, but not for source
                word = word if word in target_indexer.d else target_indexer.UNK
                if chars == 1:
                    char = [char_indexer.BOS] + list(word) + [char_indexer.EOS]
                    if len(char) > max_word_l:
                        char = char[:max_word_l]
                        char[-1] = char_indexer.EOS
                    char_idx = char_indexer.convert_sequence(pad(char, max_word_l, char_indexer.PAD))
                    targ_char.append(char_idx)

            targ = target_indexer.convert_sequence(targw)
            targ = np.array(targ, dtype=int)

            srcw = pad(srcw, newseqlength, src_indexer.PAD)
            src_char = []
            for word in srcw:
                if chars == 1:
                    word = char_indexer.clean(word)
                    char = [char_indexer.BOS] + list(word) + [char_indexer.EOS]
                    if len(char) > max_word_l:
                        char = char[:max_word_l]
                        char[-1] = char_indexer.EOS
                    char_idx = char_indexer.convert_sequence(pad(char, max_word_l, char_indexer.PAD))
                    src_char.append(char_idx)

            src = src_indexer.convert_sequence(srcw)
            src = np.array(src, dtype=int)

            align=[]

            if alignfile_hdl:
                align=alignfile_hdl.readline().strip().split(" ")
                if not alignpattern == '':
	                protectsrc = []
	                protecttarg = []
	                for i in xrange(len(src)):
	                    srcword=srcw[i]
	                    if src[i] == 2:
	                        srcword = src_indexer.UNK
	                    protectsrc.append(not(alignpattern_re.match(srcword)==None))
	                for i in xrange(len(targ)):
	                    targword=targw[i]
	                    if targ[i] == 2:
	                        targword = target_indexer.UNK
	                    protecttarg.append(not(alignpattern_re.match(targword)==None))
	                keep = True
	                for pair in align:
	                    srcidx,targidx=pair.split("-")
	                    srcword = srcw[int(srcidx)+1]
	                    targword = targw[int(targidx)+1]
	                    if src[int(srcidx)+1] == 2:
	                        srcword = "<unk>"
	                    if targ[int(targidx)+1] == 2:
	                        targword = "<unk>"
	                    if srcword == targword:
	                        # at least one match validating alignment
	                        protectsrc[int(srcidx)+1]=False
	                        protecttarg[int(targidx)+1]=False
	                if sum(protecttarg) or sum(protectsrc):
	                    if verbose: print ("DROP ALIGN\t"+src_orig.strip()+"\t"+targ_orig).encode("utf-8")
	                    dropped += 1
	                    dropped_align += 1
	                    continue

            if unkfilter > 0:
                targ_unks = float((targ[:-1] == 2).sum())
                src_unks = float((src == 2).sum())
                if unkfilter < 1: #unkfilter is a percentage if < 1
                    targ_unks = targ_unks/(len(targ[:-1])-2)
                    src_unks = src_unks/(len(src)-2)
                if targ_unks > unkfilter or src_unks > unkfilter:
                    dropped += 1
                    dropped_unk += 1
                    if verbose: print "DROP UNK\t"+src_orig.encode("utf-8")+"\n"+targ_orig.encode("utf-8").strip()+"\t"
                    continue

            targets[sent_id] = np.array(targ[:-1],dtype=int)
            target_lengths[sent_id] = (targets[sent_id] != 1).sum()
            if chars == 1:
                targets_char[sent_id] = np.array(targ_char[:-1], dtype=int)
            target_output[sent_id] = np.array(targ[1:],dtype=int)
            sources[sent_id] = np.array(src, dtype=int)
            source_lengths[sent_id] = (sources[sent_id] != 1).sum()
            if chars == 1:
                sources_char[sent_id] = np.array(src_char, dtype=int)

            source_features = load_features(src_orig_features, src_feature_indexers, newseqlength)
            target_features = load_features(targ_orig_features, target_feature_indexers, newseqlength+1)

            for i in range(len(target_feature_indexers)):
                targets_features[i][sent_id] = np.array(target_features[i][:-2], dtype=int)
                targets_features_output[i][sent_id] = np.array(target_features[i][1:-1], dtype=int)
            for i in range(len(src_feature_indexers)):
                sources_features[i][sent_id] = np.array(source_features[i][1:], dtype=int)

            if alignfile_hdl and storealign:
                for pair in align:
                    aFrom, aTo = pair.split('-')
                    aFrom = int(aFrom)
                    aTo = int(aTo)
                    alignments[sent_id][aFrom + 1][aTo + 1] = 1

            sent_id += 1
            if sent_id % 100000 == 0:
                print("{}/{} sentences processed".format(sent_id, num_sents))

        print(sent_id, num_sents)
        if shuffle == 1:
            rand_idx = np.random.permutation(sent_id)
            targets = targets[rand_idx]
            target_output = target_output[rand_idx]
            sources = sources[rand_idx]
            if alignments is not None:
                alignments = alignments[rand_idx]
            source_lengths = source_lengths[rand_idx]
            target_lengths = target_lengths[rand_idx]
            for i in range(len(sources_features)):
                sources_features[i] = sources_features[i][rand_idx]
            for i in range(len(targets_features)):
                targets_features[i] = targets_features[i][rand_idx]
                targets_features_output[i] = targets_features_output[i][rand_idx]
            if chars==1:
                sources_char = sources_char[rand_idx]
                targets_char = targets_char[rand_idx]

        #break up batches based on source lengths
        source_lengths = source_lengths[:sent_id]
        source_sort = np.argsort(source_lengths)

        sources = sources[source_sort]
        targets = targets[source_sort]
        target_output = target_output[source_sort]
        if alignments is not None:
            alignments = alignments[source_sort]
        target_l = target_lengths[source_sort]
        source_l = source_lengths[source_sort]

        for i in range(len(src_feature_indexers)):
            sources_features[i] = sources_features[i][source_sort]
        for i in range(len(target_feature_indexers)):
            targets_features[i] = targets_features[i][source_sort]
            targets_features_output[i] = targets_features_output[i][source_sort]

        curr_l = 0
        l_location = [] #idx where sent length changes

        for j,i in enumerate(source_sort):
            if source_lengths[i] > curr_l:
                curr_l = source_lengths[i]
                l_location.append(j+1)
        l_location.append(len(sources))

        #get batch sizes
        curr_idx = 1
        batch_idx = [1]
        nonzeros = []
        batch_l = []
        batch_w = []
        target_l_max = []
        for i in range(len(l_location)-1):
            while curr_idx < l_location[i+1]:
                curr_idx = min(curr_idx + batchsize, l_location[i+1])
                batch_idx.append(curr_idx)
        for i in range(len(batch_idx)-1):
            batch_l.append(batch_idx[i+1] - batch_idx[i])
            batch_w.append(source_l[batch_idx[i]-1])
            nonzeros.append((target_output[batch_idx[i]-1:batch_idx[i+1]-1] != 1).sum().sum())
            target_l_max.append(max(target_l[batch_idx[i]-1:batch_idx[i+1]-1]))

        # Write output
        f = h5py.File(outfile, "w")

        f["source"] = sources
        f["target"] = targets
        f["target_output"] = target_output
        if alignments is not None:
            print "build alignment structure"
            alignment_cc_val = []
            alignment_cc_colidx = []
            alignment_cc_sentidx = []
            S={}
            for k in range(sent_id-1):
                alignment_cc_sentidx.append(len(alignment_cc_colidx))
                for i in xrange(0, source_l[k]):
                    # for word i, build aligment vector as a string for indexing
                    a=''
                    maxnalign=0
                    # build a string representing the alignment vector
                    for j in xrange(0, newseqlength):
                        a=a+chr(ord('0')+int(alignments[k][i][j]))
                    # check if we have already built such column
                    if not a in S:
                        alignment_cc_colidx.append(len(alignment_cc_val))
                        S[a]=len(alignment_cc_val)
                        for j in xrange(0, newseqlength):
                            alignment_cc_val.append(alignments[k][i][j])
                    else:
                        alignment_cc_colidx.append(S[a])

            assert(len(alignment_cc_colidx)<4294967296)
            f["alignment_cc_sentidx"] = np.array(alignment_cc_sentidx, dtype=np.uint32)
            f["alignment_cc_colidx"] = np.array(alignment_cc_colidx, dtype=np.uint32)
            f["alignment_cc_val"] = np.array(alignment_cc_val, dtype=np.uint8)

        f["target_l"] = np.array(target_l_max, dtype=int)
        f["target_l_all"] = target_l
        f["batch_l"] = np.array(batch_l, dtype=int)
        f["batch_w"] = np.array(batch_w, dtype=int)
        f["batch_idx"] = np.array(batch_idx[:-1], dtype=int)
        f["target_nonzeros"] = np.array(nonzeros, dtype=int)
        f["source_size"] = np.array([len(src_indexer.d)])
        f["target_size"] = np.array([len(target_indexer.d)])
        f["num_source_features"] = np.array([len(src_feature_indexers)])
        f["num_target_features"] = np.array([len(target_feature_indexers)])
        f["identifier_max_size"] = np.array([int_max_size])
        for i in range(len(src_feature_indexers)):
            f["source_feature_" + str(i+1)] = sources_features[i]
            f["source_feature_" + str(i+1) + "_size"] = np.array([len(src_feature_indexers[i].d)])
            f["source_feature_" + str(i+1) + "_max_values"] = np.array([src_feature_indexers[i].max_num_values])
        for i in range(len(target_feature_indexers)):
            f["target_feature_" + str(i+1)] = targets_features[i]
            f["target_feature_output_" + str(i+1)] = targets_features_output[i]
            f["target_feature_" + str(i+1) + "_size"] = np.array([len(target_feature_indexers[i].d)])
            f["target_feature_" + str(i+1) + "_max_values"] = np.array([target_feature_indexers[i].max_num_values])
        if chars == 1:
            del sources, targets, target_output
            sources_char = sources_char[source_sort]
            f["source_char"] = sources_char
            del sources_char
            targets_char = targets_char[source_sort]
            f["target_char"] = targets_char
            f["char_size"] = np.array([len(char_indexer.d)])
        print("Saved {} sentences (dropped {} due to length [{}]/unk [{}]/align [{}] filter)".format(
            len(f["source"]), dropped, dropped_length, dropped_unk, dropped_align))
        f.close()
        return max_sent_l

    print("First pass through data to get vocab...")
    max_word_l, num_sents_train = make_vocab(args.srcfile, args.targetfile,
                                             args.seqlength, 0, args.chars)

    print("Number of sentences in training: {}".format(num_sents_train))
    max_word_l, num_sents_valid = make_vocab(args.srcvalfile, args.targetvalfile,
                                             args.seqlength, max_word_l, args.chars, 0)
    print("Number of sentences in valid: {}".format(num_sents_valid))
    if args.chars == 1:
        print("Max word length (before cutting): {}".format(max_word_l))
        max_word_l = min(max_word_l, args.maxwordlength)
        print("Max word length (after cutting): {}".format(max_word_l))

    #prune and write vocab
    src_indexer.prune_vocab(args.srcvocabsize)
    target_indexer.prune_vocab(args.targetvocabsize)
    if args.srcvocabfile != '':
        print('Loading pre-specified source vocab from ' + args.srcvocabfile)
        src_indexer.load_vocab(args.srcvocabfile, args.chars)
    if args.targetvocabfile != '':
        print('Loading pre-specified target vocab from ' + args.targetvocabfile)
        target_indexer.load_vocab(args.targetvocabfile, args.chars)
    if args.charvocabfile != '':
        print('Loading pre-specified char vocab from ' + args.charvocabfile)
        char_indexer.load_vocab(args.charvocabfile, args.chars)

    src_indexer.write(args.outputfile + ".src.dict", args.chars)
    target_indexer.write(args.outputfile + ".targ.dict", args.chars)
    if args.chars == 1:
        if args.charvocabfile == '':
            char_indexer.prune_vocab(500)
        char_indexer.write(args.outputfile + ".char.dict", args.chars)
        print("Character vocab size: {}".format(len(char_indexer.vocab)))

    if args.reusefeaturefile != '':
        load_features('source', src_feature_indexers, args.reusefeaturefile)
        load_features('target', target_feature_indexers, args.reusefeaturefile)

    save_features('source', src_feature_indexers, args.outputfile)
    save_features('target', target_feature_indexers, args.outputfile)

    print("Source vocab size: Original = {}, Pruned = {}".format(len(src_indexer.vocab),
                                                          len(src_indexer.d)))
    print("Target vocab size: Original = {}, Pruned = {}".format(len(target_indexer.vocab),
                                                          len(target_indexer.d)))

    max_sent_l = 0
    max_sent_l = convert(args.srcvalfile, args.targetvalfile, args.alignvalfile, args.alignpattern, args.storealign,
                         args.batchsize, args.seqlength,
                         args.outputfile + "-val.hdf5", num_sents_valid,
                         max_word_l, max_sent_l, args.chars, args.unkfilter, args.shuffle, args.verbose)
    max_sent_l = convert(args.srcfile, args.targetfile, args.alignfile, args.alignpattern, args.storealign,
                         args.batchsize, args.seqlength,
                         args.outputfile + "-train.hdf5", num_sents_train, max_word_l,
                         max_sent_l, args.chars, args.unkfilter, args.shuffle, args.verbose)

    print("Max sent length (before dropping): {}".format(max_sent_l))

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--srcvocabsize', help="Size of source vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                " Rest are replaced with special UNK tokens.",
                                                type=int, default=50000)
    parser.add_argument('--targetvocabsize', help="Size of target vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                "Rest are replaced with special UNK tokens.",
                                                type=int, default=50000)
    parser.add_argument('--srcfile', help="Path to source training data, "
                                           "where each line represents a single "
                                           "source/target sequence.", required=True)
    parser.add_argument('--targetfile', help="Path to target training data, "
                                           "where each line represents a single "
                                           "source/target sequence.", required=True)
    parser.add_argument('--srcvalfile', help="Path to source validation data.", required=True)
    parser.add_argument('--targetvalfile', help="Path to target validation data.", required=True)
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=64)
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer "
                                               "than this are dropped.", type=int, default=50)
    parser.add_argument('--outputfile', help="Prefix of the output file names. ", type=str, required=True)
    parser.add_argument('--maxwordlength', help="For the character models, words are "
                                           "(if longer than maxwordlength) or zero-padded "
                                            "(if shorter) to maxwordlength", type=int, default=35)
    parser.add_argument('--chars', help="If 1, construct the character-level dataset as well. "
                                        "This might take up a lot of space depending on your data "
                                        "size, so you may want to break up the training data into "
                                        "different shards.", type=int, default=0)
    parser.add_argument('--srcvocabfile', help="If working with a preset vocab, "
                                          "then including this will ignore srcvocabsize and use the"
                                          "vocab provided here.",
                                          type = str, default='')
    parser.add_argument('--targetvocabfile', help="If working with a preset vocab, "
                                         "then including this will ignore targetvocabsize and "
                                         "use the vocab provided here.",
                                          type = str, default='')
    parser.add_argument('--charvocabfile', help="If working with a preset vocab, "
                                         "then including this use the char vocab provided here.",
                                          type = str, default='')
    parser.add_argument('--unkfilter', help="Ignore sentences with too many UNK tokens. "
                                       "Can be an absolute count limit (if > 1) "
                                       "or a proportional limit (0 < unkfilter < 1).",
                                          type = float, default = 0)
    parser.add_argument('--reusefeaturefile', help="use existing feature vocabs",
                                          type = str, default ='')
    parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on  "
                                           "source length).",
                                          type = int, default = 0)
    parser.add_argument('--alignfile', help="Path to source-to-target alignment of training data, "
                                           "where each line represents a set of alignments "
                                           "per train instance.",
                                           type = str, required=False, default='')
    parser.add_argument('--alignvalfile', help="Path to source-to-target alignment of validation data",
                                           type = str, required=False, default='')
    parser.add_argument('--alignpattern', help="regular expression of patterns to align; "
                                          "if set, use alignment file to filter out sentence with mis-aligned patterns in train and validation corpora",
                                          type = str, default = r'^<unk>$')
    parser.add_argument('--storealign', help="If 1, store alignment information in hdf5 (e.g. for guided alignment)",
                                          type = int, required=False, default=0)
    parser.add_argument('--verbose', help="If 1, dump all dropped sentences",
                                          type = int, required=False, default=0)



    args = parser.parse_args(arguments)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
