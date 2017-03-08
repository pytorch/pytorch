from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import json


class IntrospectVisData():

    def __init__(self, inputs, model_name, first_outputs, meta_info, lab_arr):
        self.inputs = inputs
        self.model_name = model_name
        self.max_num_instances = inputs['num_instances']
        self.count = 0
        self.instances = []
        self.labels = [{"id": i, "name": c} for i, c in enumerate(lab_arr)]
        self.is_multilabel = True if 'multi_label' in meta_info\
                             and meta_info['multi_label'] else False

        self.conv_groups = meta_info['conv_output_names']\
                           if 'conv_output_names' in meta_info else []
        self.neuron_groups = [{"idx": i, "name": b,
                               "size": len(first_outputs[2][i -
                                           len(meta_info['output_names'])][0])}
                              for i, b in enumerate(meta_info['output_names'])]
        self.selections = [
            {"id": i, "label": "Class " + c, "type": "class"}
            for i, c in enumerate(lab_arr)]
        for i, sel in enumerate(inputs['phrase_filters']):
            self.selections.append({
                "id": i + len(self.labels), "label": sel, "type": "user"})

        self.summaries = map(lambda x: np.array([[0. for _ in range(x['size'])]
                                                 for _ in range(len(self.selections))]),
                             self.neuron_groups)

    def getInstanceActivations(self, outputs):
        outputs = outputs[(-1) * len(self.neuron_groups):]
        return [[round(_val, 3) for _val in out[0]] for out in outputs]

    def getInstanceConvActivations(self, outputs):
        outputs = outputs[(-1) * (len(self.neuron_groups) +
                          len(self.conv_groups)):(-1) * len(self.neuron_groups)]
        return [np.round(out.astype(np.float64), decimals=2).tolist()
                for out in outputs]

    def updateNeuronSummaries(self, activations, true_idxs, model_specific):
        self.count += 1
        for out_idx in range(len(self.summaries)):
            if self.is_multilabel:
                for true_idx in true_idxs:
                    self.summaries[out_idx][true_idx] += activations[out_idx]
            else:
                self.summaries[out_idx][true_idxs] += activations[out_idx]

            if "text" in model_specific:
                text = model_specific["text"]
                for sel_idx, user_sel in enumerate(self.inputs['phrase_filters']):
                    if user_sel in text:
                        self.summaries[out_idx][sel_idx + len(self.labels)] +=\
                            activations[out_idx]

    def appendInstance(self, instance):
        self.instances.append(instance)

    def processInstance(self, idx, labels, scores, outputs, model_specific):
        activations = []
        convActivations = None
        if self.model_name in ['DocNN']:
            activations = self.getInstanceActivations(outputs)
            convActivations = self.getInstanceConvActivations(outputs)
            self.updateNeuronSummaries(activations, labels, model_specific)
        if idx < self.max_num_instances:
            if len(activations) == 0:
                activations = self.getInstanceActivations(outputs),
            instance = {
                "id": idx,
                "labels": labels,
                "scores": scores,
                "activations": activations,
                "convout": convActivations,
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
            for inst_vals in inst_sort_vals:
                for i, r in enumerate(sorted(inst_vals, key=lambda x: x['val'],
                                             reverse=True)):
                    self.instances[r['inst_id']]['position'] = i

    def postprocess(self, filepath):
        self.neuron_summaries = [np.around((np.swapaxes(_s, 0, 1) /
                                           float(self.count)), 4).tolist()
                                 for _s in self.summaries] if self.count > 0 else None

        self.updateArrangements()

        with open(self.inputs['vis_file'], 'w') as vf:
            json.dump({
                "model_type": self.model_name,
                "neuron_groups": self.neuron_groups,
                "conv_groups": self.conv_groups,
                "selections": self.selections,
                "classes": self.labels,
                "instances": self.instances,
                "neuron_summaries": self.neuron_summaries,
            }, vf)
