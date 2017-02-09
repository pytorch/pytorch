from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import json


class IntrospectVisData():

    def __init__(self, model_name, first_outputs, meta_info, lab_arr):
        self.model_name = model_name
        self.max_num_instances = 1000
        self.count = 0
        self.instances = []
        self.labels = [{"id": i, "name": c} for i, c in enumerate(lab_arr)]
        self.is_multilabel = True if 'multi_label' in meta_info\
                             and meta_info['multi_label'] else False

        self.neuron_groups = [{"idx": i, "name": b,
                               "size": len(first_outputs[2][i -
                                           len(meta_info['output_names'])][0])}
                              for i, b in enumerate(meta_info['output_names'])]
        self.summaries = map(lambda x: np.array([[0. for _ in range(x['size'])]
                                                 for _ in range(len(self.labels))]),
                             self.neuron_groups)

    def getInstanceActivations(self, outputs):
        outputs = outputs[(-1) * len(self.neuron_groups):]
        return [[round(_val, 4) for _val in out[0]] for out in outputs]

    def updateNeuronSummaries(self, activations, true_idxs):
        self.count += 1
        for out_idx in range(len(self.summaries)):
            if self.is_multilabel:
                for true_idx in true_idxs:
                    self.summaries[out_idx][true_idx] += activations[out_idx]
            else:
                self.summaries[out_idx][true_idxs] += activations[out_idx]

    def appendInstance(self, instance):
        self.instances.append(instance)

    def processInstance(self, idx, labels, scores, outputs, model_specific):
        activations = []
        if self.model_name in ['DocNN']:
            activations = self.getInstanceActivations(outputs)
            self.updateNeuronSummaries(activations, labels)
        if idx < self.max_num_instances:
            if len(activations) == 0:
                activations = self.getInstanceActivations(outputs),
            instance = {
                "id": idx,
                "labels": labels,
                "scores": scores,
                "activations": activations,
            }
            for key, val in model_specific.items():
                instance[key] = val
            self.appendInstance(instance)

    def updateArrangements(self):
        if self.model_name in ['DocNN']:
            # sort class scores based on score values
            for instance in self.instances:
                instance['scores'] =\
                    sorted([{"class_id": j, "score": round(_s, 3)}
                            for j, _s in enumerate(instance['scores'])],
                           key=lambda x: x['score'], reverse=True)
            # instance positions based on scores
            inst_sort_vals = [[] for _ in range(len(self.labels))]
            for i, x in enumerate(self.instances):
                sort_val = 1.0
                # if multi_label, get the first label
                label = x['labels'] if type(x['labels']) == int\
                                    else x['labels'][0]
                if label == x['scores'][0]['class_id']:
                    # How much score difference from that of rank 2 class
                    sort_val = x['scores'][0]['score'] - x['scores'][1]['score']
                else:
                    # How much score difference from that of rank 1 class
                    sort_val = x['scores'][label]['score'] -\
                        x['scores'][0]['score']
                inst_sort_vals[label].append({"inst_id": i, "val": sort_val})
            for class_id, inst_vals in enumerate(inst_sort_vals):
                for i, r in enumerate(sorted(inst_vals, key=lambda x: x['val'],
                                             reverse=True)):
                    self.instances[r['inst_id']]['position'] = i

    def postprocess(self, filepath):
        self.neuron_summaries = [np.around((np.swapaxes(_s, 0, 1) /
                                           float(self.count)), 4).tolist()
                                 for _s in self.summaries] if self.count > 0 else None

        self.updateArrangements()

        with open(filepath, 'w') as vf:
            json.dump({
                "model_type": self.model_name,
                "neuron_groups": self.neuron_groups,
                "classes": self.labels,
                "instances": self.instances,
                "neuron_summaries": self.neuron_summaries,
            }, vf)
