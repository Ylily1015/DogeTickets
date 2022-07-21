# -*- coding: utf-8 -*-

import os
import csv
import collections
import json


Example = collections.namedtuple(
    "Example", 
    (
        "uid", 
        "text_a", 
        "text_b", 
        "label",
        "domain"
    )
)


class DataReader:
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.tsv")), 
            "train",
        )

    def get_dev_examples(self):
        return self._create_examples(
        self._read_tsv(os.path.join(self.data_dir, "dev.tsv")), 
            "dev",
        )

    def get_test_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.tsv")), 
            "test",
        )

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
            
    @staticmethod      
    def _read_json(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            #num = 0 
            for line in f.readlines():
                #if(num==600):
                    #break
                #num+=1
                data = json.loads(line.strip())
                lines.append(data)
            return lines

    @staticmethod
    def get_label_map():
        """Gets the label map for this data set."""
        raise NotImplementedError()

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples."""
        raise NotImplementedError()

class MNLIReader(DataReader):
    """Reader for the MultiNLI data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_train_examples(self, domains):
        examples = []
        for domain in domains:
            train_dir = os.path.join(self.data_dir, domain)
            examples.extend(
                self._create_examples(
                    self._read_json(os.path.join(train_dir, "train.json")), "train", domain=domain)
            )
        return examples

    def get_dev_examples(self, domains):
        examples = []
        for domain in domains:
            dev_dir = os.path.join(self.data_dir, domain)
            examples.extend(
                self._create_examples(
                    self._read_json(os.path.join(dev_dir, "dev.json")), "dev", domain=domain)
            )
        return examples

    def get_test_examples(self, domains):
        examples = []
        for domain in domains:
            test_dir = os.path.join(self.data_dir, domain)
            examples.extend(
                self._create_examples(
                    self._read_json(os.path.join(test_dir, "test.json")), "test", domain=domain)
            )
        return examples

    @staticmethod
    def get_label_map():
        d = {"contradiction": 0, "entailment": 1 ,"neutral": 2}
        r_d = {v:k for k,v in d.items()}
        return lambda x: d[x], lambda x: r_d[x], len(d)

    @staticmethod
    def get_domain_map():
        d = {
            "fiction": 0,
            "government": 1, 
            "slate": 2, 
            "telephone": 3,
            "travel": 4,
            "facetoface": 5, 
            "letters": 6, 
            "nineeleven": 7, 
            "oup": 8, 
            "verbatim": 9
        }
        return lambda x: d[x]

    @staticmethod
    def _create_examples(lines, set_type, domain=None):
        examples = []
        
        for (i, line) in enumerate(lines):
            
            uid = "%s-%s" % (set_type, i)
            
            text_a = line["text_a"]
            text_b = line["text_b"]
            label = str(line["label"])
            
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label,
                    domain=domain
                )
            )
        return examples

class ARDReader(DataReader):
    """Reader for the Amazon Review data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_train_examples(self, domains):
        examples = []
        for domain in domains:
            train_dir = os.path.join(self.data_dir, domain)
            examples.extend(
                self._create_examples(
                    self._read_json(os.path.join(train_dir, "train.json")), "train", domain=domain)
            )
        return examples

    def get_dev_examples(self, domains):
        examples = []
        for domain in domains:
            dev_dir = os.path.join(self.data_dir, domain)
            examples.extend(
                self._create_examples(
                    self._read_json(os.path.join(dev_dir, "dev.json")), "dev", domain=domain)
            )
        return examples

    def get_test_examples(self, domains):
        examples = []
        for domain in domains:
            test_dir = os.path.join(self.data_dir, domain)
            examples.extend(
                self._create_examples(
                    self._read_json(os.path.join(test_dir, "test.json")), "test", domain=domain)
            )
        return examples

    @staticmethod
    def get_label_map():
        d = {"0": 0, "1": 1 ,"2": 2}
        r_d = {v:k for k,v in d.items()}
        return lambda x: d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def get_domain_map():
        d = {
            "All_Beauty": 0, 
            "Automotive": 1,
            "Digital_Music": 2,
            "Gift_Cards": 3,
            "Industrial_and_Scientific": 4,
            "Movies_and_TV": 5,
            "Software": 6
        }
        return lambda x: d[x]

    @staticmethod
    def _create_examples(lines, set_type, domain=None):
        examples = []
        
        for (i, line) in enumerate(lines):
            
            uid = "%s-%s" % (set_type, i)
            
            text_a = line["text"]
            label = str(line["label"])
            
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=None, 
                    label=label,
                    domain=domain
                )
            )
        return examples

class OntoNoteReader(DataReader):
    """Reader for the OntoNote data set ."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_train_examples(self, domains):
        examples = []
        for domain in domains:
            train_dir = os.path.join(self.data_dir, domain)
            examples.extend(
                self._create_examples(
                    self._read_json(os.path.join(train_dir, "train.json")), "train", domain=domain)
            )
        return examples

    def get_dev_examples(self, domains):
        examples = []
        for domain in domains:
            dev_dir = os.path.join(self.data_dir, domain)
            examples.extend(
                self._create_examples(
                    self._read_json(os.path.join(dev_dir, "dev.json")), "dev", domain=domain)
            )
        return examples

    def get_test_examples(self, domains):
        examples = []
        for domain in domains:
            test_dir = os.path.join(self.data_dir, domain)
            examples.extend(
                self._create_examples(
                    self._read_json(os.path.join(test_dir, "test.json")), "test", domain=domain)
            )
        return examples

    @staticmethod
    def get_label_map():
        d={"O": 0, "I-PERSON": 1, "B-PERSON": 2, "I-WORK_OF_ART": 3, "B-DATE": 4,"B-CARDINAL": 5, 
           "I-LOC": 6, "I-QUANTITY": 7, "B-PERCENT": 8, "I-CARDINAL": 9, "B-LANGUAGE": 10, "I-ORDINAL": 11, 
           "B-WORK_OF_ART": 12, "B-ORDINAL": 13, "I-DATE": 14, "B-MONEY": 15, "B-LAW": 16, "B-GPE": 17, 
           "B-PRODUCT": 18, "B-TIME": 19, "B-QUANTITY": 20, "I-MONEY": 21, "B-ORG": 22, "B-FAC": 23, 
           "I-EVENT": 24, "B-LOC": 25, "I-LAW": 26, "I-NORP": 27, "I-ORG": 28, "I-GPE": 29, "B-EVENT": 30, 
           "I-PERCENT": 31, "I-PRODUCT": 32, "B-NORP": 33, "I-TIME": 34, "I-FAC": 35, "I-LANGUAGE":36}
        r_d = {v:k for k,v in d.items()}
        return lambda x: d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def get_domain_map():
        d = {
            "bc": 0, 
            "bn": 1,
            "mz": 2,
            "nw": 3,
            "pt": 4,
            "tc": 5,
            "wb": 6
        }
        return lambda x: d[x]

    @staticmethod
    def _create_examples(lines, set_type, domain=None):
        examples = []
        
        for (i, line) in enumerate(lines):
            
            uid = "%s-%s" % (set_type, i)
            
            text_a = line["text"]
            label = line["label"]
            
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=None, 
                    label=label,
                    domain=domain
                )
            )
        return examples
