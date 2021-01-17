import math
import operator
import numpy as np
import sys
"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

tags = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'IN', 'NUM', 'PART', 'UH', 'X', 'MODAL', 'CONJ', 'PERIOD', 'PUNCT', 'TO', 'START'}

def baseline(train, test):
    predicts = []
    word_to_tag_counts = {}
    tag_totals = {}

    for sentence in train:
        for pair in sentence:
            word, tag = pair

            if word not in word_to_tag_counts:
                word_to_tag_counts[word] = {}
            if tag in word_to_tag_counts[word]:
                word_to_tag_counts[word][tag] += 1
            else:
                word_to_tag_counts[word][tag] = 1

            if tag in tag_totals:
                tag_totals[tag] += 1
            else:
                tag_totals[tag] = 1


    for sentence in test:
        sentence_prediction = []
        for word in sentence:
            if word in word_to_tag_counts:
                tag_map = word_to_tag_counts[word]
                best_tag = getMaxTag(tag_map)
                sentence_prediction.append((word, best_tag))
            else:
                sentence_prediction.append((word, getMaxTag(tag_totals)))
        predicts.append(sentence_prediction)

    return predicts


def getMaxTag(tag_map):
    return max(tag_map.keys(), key=(lambda key: tag_map[key]))

def get_counts(train):

    transition_seq = {}
    tag_counts = {}
    tag_seq = {}
    emission_dist = {}
    word_counts = {}
    hapax_list = {}


    for sentence in train:
        prev_tag = 'START'

        if sentence[0][1] not in tag_seq.keys():
            tag_seq[sentence[0][1]] = 1
        else:
            tag_seq[sentence[0][1]] += 1

        for pair in sentence:
            word, tag = pair
            if tag not in tag_counts.keys():
                tag_counts[tag] = 1
            else:
                tag_counts[tag] += 1

            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1    

            if word not in emission_dist.keys():
                emission_dist[word] = {tag : 1}
            elif word in emission_dist.keys():
                emission_dist[word][tag] = emission_dist[word].get(tag, 0) + 1

            if tag not in transition_seq.keys():
                transition_seq[tag] = {prev_tag : 1}
            elif tag in transition_seq.keys():
                transition_seq[tag][prev_tag] = transition_seq[tag].get(prev_tag, 0) + 1
            prev_tag = tag

    for word in word_counts.keys():
        if word_counts[word] == 1:
            hapax_list[word] = emission_dist[word]

    return tag_counts, tag_seq, transition_seq, emission_dist, hapax_list

class Viterbi:
    def __init__(self, probability, word, previous_tag, current_tag, previous_tag_idx):
        """Return a Node Object"""
        self.probability = probability
        self.word = word
        self.previous_tag = previous_tag
        self.current_tag = current_tag
        self.previous_tag_idx = previous_tag_idx

    def __hash__(self):
        return hash(self.probability)

    def __eq__(self, other):
        return self.probability == other.probability

    def __lt__(self, other):
        return self.probability < other.probability

def get_transition_probs(transition_dist, tag_counts, smoothing_parameter):

    transition_probs = {}

    for first_tag in tag_counts.keys():
        for second_tag in tag_counts.keys():      
            probability = math.log((transition_dist.get(first_tag, 0).get(second_tag, 0) + smoothing_parameter)/(tag_counts[second_tag] + smoothing_parameter * (len(tag_counts.keys()) + 1)))

            if first_tag not in transition_probs.keys():
                transition_probs[first_tag] = {second_tag : probability}
            elif first_tag in transition_probs.keys():
                transition_probs[first_tag][second_tag] = probability

    return transition_probs

def get_emission_probs(emission_dist, tag_counts, hapax_prob, smoothing_parameter):

    emission_probs = {}

    for word in emission_dist.keys():
        for tag in tag_counts.keys():
            numerator = emission_dist[word].get(tag, 0) + smoothing_parameter
            denominator = tag_counts[tag] + smoothing_parameter * (len(tag_counts.keys()) + 1)

            if word not in emission_probs.keys():
                emission_probs[word] = {tag : math.log(numerator/denominator)}
            elif word in emission_probs.keys():
                emission_probs[word][tag] = math.log(numerator/denominator)

            if 'UNKNOWN' not in emission_probs.keys():
                numerator = smoothing_parameter * hapax_prob[tag]
                denominator = tag_counts[tag] + (smoothing_parameter * hapax_prob[tag]) * (len(tag_counts.keys()) + 1)

                emission_probs['UNKNOWN'] = {tag : math.log(numerator/denominator)}
            elif 'UNKNOWN' in emission_probs.keys():
                emission_probs['UNKNOWN'][tag] = math.log((smoothing_parameter * hapax_prob[tag])/(tag_counts[tag] + (smoothing_parameter * hapax_prob[tag]) * (len(tag_counts.keys()) + 1)))

    return emission_probs

def viterbi_helper(sentence, tag_counts, initial_prob, transition_prob, emission_prob):

    trellis = []
    prediction = []

    for tag in tag_counts.keys():
        temp = []
        for word in sentence:
            temp.append(Viterbi(-sys.maxsize - 1, word, 'NONE', 'NONE', 'NONE'))
        trellis.append(temp)

    for idx, tag in enumerate(tag_counts.keys()):
        if sentence[0] in emission_prob.keys():
            initialization = initial_prob[tag] + emission_prob[sentence[0]][tag]
        else:
            initialization = initial_prob[tag] + emission_prob['UNKNOWN'][tag]
        trellis[idx][0].probability = initialization
        trellis[idx][0].previous_tag = 'NONE'
        trellis[idx][0].current_tag = tag
        trellis[idx][0].previous_tag_idx = 'NONE'


    max_prob = 0 
    previous_tag = ' ' 
    current_tag = ' '
    previous_tag_idx = ' '
    

    for idx, word in enumerate(sentence):
        if idx > 0:
            for tag_idx, curr_tag in enumerate(tag_counts.keys()):
                compare_to = -sys.maxsize - 1
                for prev_tag_idx, prev_tag in enumerate(tag_counts.keys()):
                    if word in emission_prob.keys():
                        temp_viterbi = trellis[prev_tag_idx][idx - 1].probability + transition_prob[curr_tag][prev_tag] + emission_prob[word][curr_tag]
                    else:
                        temp_viterbi = trellis[prev_tag_idx][idx - 1].probability + transition_prob[curr_tag][prev_tag] + emission_prob['UNKNOWN'][curr_tag]
                    
                    if temp_viterbi > compare_to:
                        compare_to = temp_viterbi
                        max_prob = temp_viterbi
                        current_tag = curr_tag
                        previous_tag = prev_tag
                        previous_tag_idx = prev_tag_idx


                trellis[tag_idx][idx].probability = max_prob
                trellis[tag_idx][idx].current_tag = current_tag
                trellis[tag_idx][idx].previous_tag = previous_tag
                trellis[tag_idx][idx].previous_tag_idx = previous_tag_idx

                max_prob = 0 
                previous_tag = ' ' 
                current_tag = ' '
                previous_tag_idx = ' '

    max_tag_idx = 0
    compare_to = -sys.maxsize - 1
    
    for tag_idx, tag in enumerate(tag_counts.keys()):
        if trellis[tag_idx][len(sentence) - 1].probability > compare_to:
            compare_to = trellis[tag_idx][len(sentence) - 1].probability
            max_tag_idx = tag_idx
        
    for word_idx in range(len(sentence) - 1, -1, -1):
        max_tag = trellis[max_tag_idx][word_idx].current_tag
        max_tag_idx = trellis[max_tag_idx][word_idx].previous_tag_idx
        prediction.append((sentence[word_idx], max_tag))

    return prediction[::-1]

def get_hapax(train, hapax_list, tag_counts, smoothing_parameter):
    
    tag_freq = {}
    hapax_prob = {}

    for word in hapax_list.keys():
        for tag in hapax_list[word]:
            if tag not in tag_freq.keys():
                tag_freq[tag] = 1
            else:
                tag_freq[tag] += 1
    for tag in tag_counts.keys():
        if tag in tag_freq.keys():
            hapax_prob[tag] = (tag_freq[tag] + smoothing_parameter)/(len(hapax_list) + smoothing_parameter * (len(tag_freq.keys()) + 1))    
        else:
            hapax_prob[tag] = smoothing_parameter/(len(hapax_list) + smoothing_parameter * (len(tag_freq.keys()) + 1))   

    return hapax_prob

def viterbi(train, test):
    '''
    TODO: implement the Viterbi algorithm.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_counts, tag_dists, transition_dist, emission_dist, hapax_set = get_counts(train)
    smoothing_parameter = 0.0001

    hapax_prob = get_hapax(train, hapax_set, tag_counts, smoothing_parameter)
    
    initial_probs = {}
    for tag in tag_counts.keys():
        initial_probs[tag] = math.log(tag_dists.get(tag, smoothing_parameter)/(len(train) + smoothing_parameter * (len(tag_counts.keys()) + 1)))

    transition_prob = get_transition_probs(transition_dist, tag_counts, smoothing_parameter)
    emission_prob = get_emission_probs(emission_dist, tag_counts, hapax_prob, smoothing_parameter)

    predicts = []

    for sentence in test:
        sentence_prediction = viterbi_helper(sentence, tag_counts, initial_probs, transition_prob, emission_prob)
        predicts.append(sentence_prediction)


    return predicts








