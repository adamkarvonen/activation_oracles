import csv
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset

logger = logging.getLogger(__name__)

YES_TOKEN = "Yes"
NO_TOKEN = "No"
DEFAULT_DATA_DIR = "datasets/classification_datasets"


@dataclass
class ContextQASample:
    context: str
    questions: list[str]
    answers: list[str]
    ds_label: str | None = None

    def __post_init__(self):
        for q, a in zip(self.questions, self.answers):
            assert a in (YES_TOKEN, NO_TOKEN)
            assert "#" in q


class DatasetLoader:
    def __init__(self, group, name):
        self.group = group
        self.name = name

        with open(os.path.join(DEFAULT_DATA_DIR, "paraphrases/question.json")) as f:
            self.question_paraphrases = json.load(f)[group]

    # Must be overridden by dataset class
    def load(self, num_qa_per_sample: int) -> list[ContextQASample]:
        raise NotImplementedError


class MdGenderDatasetLoader(DatasetLoader):
    GROUP_NAME = "md_gender"
    DATASET_NAME = "md_gender"

    def __init__(self):
        super().__init__(self.__class__.GROUP_NAME, self.__class__.DATASET_NAME)

    def load(self, num_qa_per_sample: int):
        dataset = load_dataset("facebook/md_gender_bias", name="funpedia", trust_remote_code=True)
        all_examples = []
        female_count = 0
        for split in ("train", "validation", "test"):
            for text, entity, gender in zip(
                dataset[split]["text"],
                dataset[split]["title"],
                dataset[split]["gender"],
            ):
                if gender == 0:
                    # skip gender-neutral
                    continue
                gender = {1: "female", 2: "male"}[gender]
                if gender == "female":
                    female_count += 1
                all_examples.append((text, entity, gender))

        # Shuffle and go through examples again to balance the labels
        random.shuffle(all_examples)

        result = []
        male_count = 0
        for text, entity, context_label in all_examples:
            if context_label == "male":
                if male_count >= female_count:
                    continue
                male_count += 1

            questions = []
            answers = []
            paraphrases = random.sample(self.question_paraphrases, num_qa_per_sample)
            for paraphrase in paraphrases:
                question_label = random.choice(["female", "male"])
                question = "# " + paraphrase.format(question_label)
                answer = YES_TOKEN if context_label == question_label else NO_TOKEN
                questions.append(question)
                answers.append(answer)

            context = f"{text}\n\nThis text is about {entity}."

            result.append(
                ContextQASample(
                    context=context,
                    questions=questions,
                    answers=answers,
                    ds_label=context_label,
                )
            )
        return result


class SnliDatasetLoader(DatasetLoader):
    GROUP_NAME = "snli"
    DATASET_NAME = "snli"

    def __init__(self):
        super().__init__(self.__class__.GROUP_NAME, self.__class__.DATASET_NAME)

    def load(self, num_qa_per_sample: int):
        print("Loading SNLI dataset")
        dataset = load_dataset("stanfordnlp/snli")["train"]
        examples = []
        for example in tqdm(dataset):
            if example["label"] not in (0, 2):
                # skip neutral
                continue

            answer = {0: YES_TOKEN, 2: NO_TOKEN}[example["label"]]

            paraphrases = random.sample(self.question_paraphrases, num_qa_per_sample)
            questions = []
            for paraphrase in paraphrases:
                question = f"# {paraphrase} {example['hypothesis']}"
                questions.append(question)

            examples.append(
                ContextQASample(
                    context=example["premise"],
                    questions=questions,
                    answers=[answer] * num_qa_per_sample,
                )
            )

        return examples


class AgNewsDatasetLoader(DatasetLoader):
    GROUP_NAME = "ag_news"
    DATASET_NAME = "ag_news"
    DATA_FILES_PATH = os.path.join(DEFAULT_DATA_DIR, "ag_news")

    def __init__(self):
        super().__init__(self.__class__.GROUP_NAME, self.__class__.DATASET_NAME)

    def load(self, num_qa_per_sample: int):
        label_to_topic = {
            "1": "World News",
            "2": "Sports",
            "3": "Business",
            "4": "Science/Technology",
        }
        labels = set(label_to_topic.keys())
        examples = []
        with open(os.path.join(self.DATA_FILES_PATH, "ag_news.csv")) as f:
            reader = csv.DictReader(f)
            for row in reader:
                correct_label = row["Class Index"]

                title = row["Title"]
                description = row["Description"]

                context = f"{title}\n\n{description}"
                questions = []
                answers = []

                paraphrases = random.sample(self.question_paraphrases, num_qa_per_sample)
                for paraphrase in paraphrases:
                    incorrect_label = random.choice(list(labels - {correct_label}))
                    question_label = random.choice((correct_label, incorrect_label))
                    question = "# " + paraphrase.format(label_to_topic[question_label])
                    answer = YES_TOKEN if question_label == correct_label else NO_TOKEN
                    questions.append(question)
                    answers.append(answer)

                examples.append(
                    ContextQASample(
                        context=context,
                        questions=questions,
                        answers=answers,
                        ds_label=label_to_topic[correct_label],
                    )
                )
        return examples


class NerDatasetLoader(DatasetLoader):
    GROUP_NAME = "ner"
    DATASET_NAME = "ner"
    DATA_FILES_PATH = os.path.join(DEFAULT_DATA_DIR, "ner")

    def __init__(self):
        super().__init__(self.__class__.GROUP_NAME, self.__class__.DATASET_NAME)

    def _get_qa_for_sentence(self, sentence, sentence_entities, all_entities, num_qa_per_sample):
        context = " ".join(sentence)
        questions = []
        answers = []

        sentence_entities_set = set(sentence_entities)
        paraphrases = random.sample(self.question_paraphrases, num_qa_per_sample)
        for paraphrase in paraphrases:
            correct_label = random.choice(sentence_entities)

            # Doing this in a loop is 2x faster than computing the set difference
            incorrect_label = correct_label
            while incorrect_label in sentence_entities_set:
                incorrect_label = random.choice(all_entities)

            question_label = random.choice((correct_label, incorrect_label))
            question = "# " + paraphrase.format(question_label)
            answer = YES_TOKEN if question_label == correct_label else NO_TOKEN
            questions.append(question)
            answers.append(answer)

        return ContextQASample(context=context, questions=questions, answers=answers)

    def load(self, num_qa_per_sample: int):
        all_sentences = []
        all_entities = set()
        print("Reading NER dataset")
        with open(os.path.join(self.DATA_FILES_PATH, "ner.csv"), encoding="unicode_escape") as f:
            reader = csv.DictReader(f)
            current_sentence = []
            sentence_entities = []
            current_entity = []
            for row in reader:
                sentence_id = row["Sentence #"]
                word = row["Word"]
                tag = row["Tag"]

                if sentence_id.strip() != "" and len(current_sentence) > 0:
                    if len(current_entity) > 0:
                        sentence_entities.append(" ".join(current_entity))
                        current_entity = []
                    all_sentences.append((current_sentence, sentence_entities))
                    current_sentence = []
                    sentence_entities = []

                current_sentence.append(word)

                if (tag == "O" or tag.startswith("B")) and len(current_entity) > 0:
                    entity = " ".join(current_entity)
                    all_entities.add(entity)
                    sentence_entities.append(" ".join(current_entity))
                    current_entity = []
                elif tag.startswith("B"):
                    current_entity = []
                    current_entity.append(word)
                elif tag.startswith("I"):
                    current_entity.append(word)

            if len(current_sentence) > 0:
                if len(current_entity) > 0:
                    entity = " ".join(current_entity)
                    all_entities.add(entity)
                    sentence_entities.append(entity)
                all_sentences.append((current_sentence, sentence_entities))

        examples = []
        print("Processing NER dataset")
        for sentence, sentence_entities in tqdm(all_sentences):
            if len(sentence_entities) == 0:
                continue
            examples.append(
                self._get_qa_for_sentence(sentence, sentence_entities, list(all_entities), num_qa_per_sample)
            )
        return examples


class GeometryOfTruthDatasetLoader(DatasetLoader):
    GROUP_NAME = "geometry_of_truth"
    DATA_FILES_PATH = os.path.join(DEFAULT_DATA_DIR, "gmt")

    DATASET_NAMES = [
        "sp_en_trans",
        # "neg_sp_en_trans",
        "cities",
        # "neg_cities",
        "smaller_than",
        "larger_than",
        # "common_claim_true_false",
        "companies_true_false",
        # "counterfact_true_false",
    ]

    def load(self, num_qa_per_sample: int):
        examples = []
        filename = self.name + ".csv"
        with open(os.path.join(self.DATA_FILES_PATH, filename)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions = []
                paraphrases = random.sample(self.question_paraphrases, num_qa_per_sample)
                for paraphrase in paraphrases:
                    question = "# " + paraphrase
                    questions.append(question)
                answer = {"0": NO_TOKEN, "1": YES_TOKEN}[row["label"]]
                answers = [answer] * num_qa_per_sample

                example = ContextQASample(
                    context=row["statement"],
                    questions=questions,
                    answers=answers,
                    ds_label=row["label"],
                )
                examples.append(example)

        return examples

    @staticmethod
    def get_all_loaders():
        loaders = []
        for name in GeometryOfTruthDatasetLoader.DATASET_NAMES:
            loaders.append(GeometryOfTruthDatasetLoader(GeometryOfTruthDatasetLoader.GROUP_NAME, name))
        return loaders


class SstDatasetLoader(DatasetLoader):
    GROUP_NAME = "sst2"
    DATASET_NAME = "sst2"

    def __init__(self):
        super().__init__(self.__class__.GROUP_NAME, self.__class__.DATASET_NAME)

    def load(self, num_qa_per_sample: int):
        dataset = load_dataset("stanfordnlp/sst2")
        result = []
        for split in ("train", "validation"):
            for sentence, label in zip(dataset[split]["sentence"], dataset[split]["label"]):
                context_label = {0: "negative", 1: "positive"}[label]
                questions = []
                answers = []
                paraphrases = {
                    label: random.sample(self.question_paraphrases[label], num_qa_per_sample)
                    for label in ("positive", "negative")
                }
                for i in range(num_qa_per_sample):
                    question_label = random.choice(["negative", "positive"])
                    question = "# " + paraphrases[question_label][i]
                    answer = YES_TOKEN if context_label == question_label else NO_TOKEN
                    questions.append(question)
                    answers.append(answer)

                result.append(
                    ContextQASample(
                        context=sentence.strip(),
                        questions=questions,
                        answers=answers,
                        ds_label=context_label,
                    )
                )
        return result


# TODO (arnab): Remove some of the poor performing relations.
RELATION_FILES_ROOT = os.path.join(DEFAULT_DATA_DIR, "relations")
RELATION_NAMES = []
for relation_type in os.listdir(RELATION_FILES_ROOT):
    for file_name in os.listdir(os.path.join(RELATION_FILES_ROOT, relation_type)):
        if file_name.endswith(".json"):
            RELATION_NAMES.append(f"{relation_type}/{file_name[:-5]}")


class RelationDatasetLoader(DatasetLoader):
    GROUP_NAME = "relations"
    DATA_FILES_PATH = RELATION_FILES_ROOT
    DATASET_NAMES = RELATION_NAMES

    def load(self, num_qa_per_sample: int):
        relation_type, relation_name = self.name.split("/")
        file_path = os.path.join(self.DATA_FILES_PATH, relation_type, f"{relation_name}.json")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []

        examples = []
        with open(file_path, "r") as f:
            data_dict = json.load(f)
            prompt_templates = data_dict["prompt_templates"]
            objects = [sample["object"] for sample in data_dict["samples"]]
            objects = set(objects)
            for sample in data_dict["samples"]:
                template = random.choice(prompt_templates) + " {}."
                questions = random.sample(self.question_paraphrases, num_qa_per_sample)
                examples.append(
                    ContextQASample(
                        context=template.format(sample["subject"], sample["object"]),
                        questions=["# " + q for q in questions],
                        answers=[YES_TOKEN] * num_qa_per_sample,
                    )
                )
                questions = random.sample(self.question_paraphrases, num_qa_per_sample)
                false_obj = random.choice(list(objects - {sample["object"]}))
                examples.append(
                    ContextQASample(
                        context=template.format(sample["subject"], false_obj),
                        questions=["# " + q for q in questions],
                        answers=[NO_TOKEN] * num_qa_per_sample,
                    )
                )
        logger.info(f"Loaded {len(examples)} examples from {self.name}.")
        return examples

    @staticmethod
    def get_all_loaders():
        loaders = []
        for name in RelationDatasetLoader.DATASET_NAMES:
            loaders.append(RelationDatasetLoader(RelationDatasetLoader.GROUP_NAME, name))
        return loaders


class TenseDatasetLoader(DatasetLoader):
    GROUP_NAME = "tense"
    DATASET_NAME = "tense"
    DATASET_PATH = os.path.join(DEFAULT_DATA_DIR, "tense", "tense_processed.json")

    def __init__(self):
        super().__init__(self.__class__.GROUP_NAME, self.__class__.DATASET_NAME)

    def load(self, num_qa_per_sample: int) -> list[ContextQASample]:
        context_qa: list[ContextQASample] = []
        with open(self.DATASET_PATH, "r") as f:
            examples = json.load(f)

        class_labels = set([sample["label"] for sample in examples])
        for example in examples:
            questions = []
            answers = []
            correct_label = example["label"]
            for idx in range(num_qa_per_sample):
                ans = random.choice([YES_TOKEN, NO_TOKEN])
                ques = "# " + random.choice(self.question_paraphrases)
                answers.append(ans)
                if ans == YES_TOKEN:
                    questions.append(ques.format(correct_label))
                else:
                    incorrect_label = random.choice(list(class_labels - {correct_label}))
                    questions.append(ques.format(incorrect_label))

            context_qa.append(
                ContextQASample(
                    context=example["sentence"],
                    questions=questions,
                    answers=answers,
                    ds_label=correct_label,
                )
            )

        return context_qa


class SingularPluralDatasetLoader(DatasetLoader):
    GROUP_NAME = "singular_plural"
    DATASET_NAME = "singular_plural"
    DATASET_PATH = os.path.join(DEFAULT_DATA_DIR, "singular_plural", "singular_plural.csv")

    def __init__(self):
        super().__init__(self.__class__.GROUP_NAME, self.__class__.DATASET_NAME)

    def load(self, num_qa_per_sample: int) -> list[ContextQASample]:
        context_qa: list[ContextQASample] = []
        with open(self.DATASET_PATH, "r") as f:
            df = pd.read_csv(f)
        examples = df.to_dict(orient="records")

        class_labels = {"single", "multiple"}
        for example in examples:
            questions = []
            answers = []
            correct_label = example["n_subjects"]
            for idx in range(num_qa_per_sample):
                ans = random.choice([YES_TOKEN, NO_TOKEN])
                incorrect_label = random.choice(list(class_labels - {correct_label}))
                answers.append(ans)
                if ans == YES_TOKEN:
                    questions.append(
                        "# " + random.choice(self.question_paraphrases[correct_label]).format(correct_label)
                    )
                else:
                    questions.append(
                        "# " + random.choice(self.question_paraphrases[incorrect_label]).format(incorrect_label)
                    )

            context_qa.append(
                ContextQASample(
                    context=example["sentence"],
                    questions=questions,
                    answers=answers,
                    ds_label=example["n_subjects"],
                )
            )

        return context_qa


class EngelsDatasetLoader(DatasetLoader):
    GROUP_NAME = "engels"
    DATA_FILES_PATH = os.path.join(DEFAULT_DATA_DIR, "engels")

    # Dataset name -> file mapping and label schema (positive, negative)
    DATASET_CONFIGS = {
        "news_class_politics": {
            "file": "139_news_class_Politics.csv",
            "labels": ("politics", "not_politics"),
        },
        "headline_istrump": {
            "file": "21_headline_istrump.csv",
            "labels": ("trump", "not_trump"),
        },
        "headline_isobama": {
            "file": "22_headline_isobama.csv",
            "labels": ("obama", "not_obama"),
        },
        "headline_ischina": {
            "file": "23_headline_ischina.csv",
            "labels": ("china", "not_china"),
        },
        "wikidata_isjournalist": {
            "file": "59_wikidata_occupation_isjournalist.csv",
            "labels": ("journalist", "not_journalist"),
        },
        "hist_fig_ismale": {
            "file": "5_hist_fig_ismale.csv",
            "labels": ("male", "female"),
        },
        "wikidata_isathlete": {
            "file": "60_wikidata_occupation_isathlete.csv",
            "labels": ("athlete", "not_athlete"),
        },
        "wikidata_ispolitician": {
            "file": "62_wikidata_occupation_ispolitician.csv",
            "labels": ("politician", "not_politician"),
        },
        "wikidata_issinger": {
            "file": "63_wikidata_occupation_issinger.csv",
            "labels": ("singer", "not_singer"),
        },
        "wikidata_isresearcher": {
            "file": "64_wikidata_occupation_isresearcher.csv",
            "labels": ("researcher", "not_researcher"),
        },
    }

    def __init__(self, name: str):
        super().__init__(self.__class__.GROUP_NAME, name)
        # Narrow paraphrases to the specific dataset within the Engels group
        with open(os.path.join(DEFAULT_DATA_DIR, "paraphrases/question.json")) as f:
            self.question_paraphrases = json.load(f)[self.__class__.GROUP_NAME][name]

    def load(self, num_qa_per_sample: int) -> list[ContextQASample]:
        cfg = self.DATASET_CONFIGS[self.name]
        df = pd.read_csv(os.path.join(self.DATA_FILES_PATH, cfg["file"]))
        examples = df.to_dict(orient="records")

        pos_label, neg_label = cfg["labels"]
        class_labels = {pos_label, neg_label}

        context_qa: list[ContextQASample] = []
        for example in examples:
            target = example["target"]
            is_positive = target in (1, "1", True, "True", "true")
            correct_label = pos_label if is_positive else neg_label

            questions = []
            answers = []
            for _ in range(num_qa_per_sample):
                ans = random.choice([YES_TOKEN, NO_TOKEN])
                answers.append(ans)
                if ans == YES_TOKEN:
                    q_label = correct_label
                else:
                    q_label = random.choice(list(class_labels - {correct_label}))
                questions.append(
                    "# "
                    + random.choice(self.question_paraphrases[q_label]).format(q_label)
                )

            context_qa.append(
                ContextQASample(
                    context=str(example["prompt"]).strip(),
                    questions=questions,
                    answers=answers,
                    ds_label=correct_label,
                )
            )

        return context_qa

    @staticmethod
    def get_all_loaders():
        return [EngelsDatasetLoader(name) for name in EngelsDatasetLoader.DATASET_CONFIGS]


class BaseEngelsCsvBinaryLoader(DatasetLoader):
    DATA_FILES_PATH = os.path.join(DEFAULT_DATA_DIR, "engels")
    GROUP_NAME: str
    DATASET_NAME: str
    FILE_NAME: str
    LABELS: tuple[str, str]

    def __init__(self):
        # Do not call super(); we want to load paraphrases from the shared 'engels' block
        self.group = self.__class__.GROUP_NAME
        self.name = self.__class__.DATASET_NAME
        with open(os.path.join(DEFAULT_DATA_DIR, "paraphrases/question.json")) as f:
            self.question_paraphrases = json.load(f)["engels"][self.__class__.DATASET_NAME]

    def load(self, num_qa_per_sample: int) -> list[ContextQASample]:
        df = pd.read_csv(os.path.join(self.DATA_FILES_PATH, self.__class__.FILE_NAME))
        examples = df.to_dict(orient="records")

        pos_label, neg_label = self.__class__.LABELS
        class_labels = {pos_label, neg_label}

        context_qa: list[ContextQASample] = []
        for example in examples:
            target = example["target"]
            is_positive = target in (1, "1", True, "True", "true")
            correct_label = pos_label if is_positive else neg_label

            questions = []
            answers = []
            for _ in range(num_qa_per_sample):
                ans = random.choice([YES_TOKEN, NO_TOKEN])
                answers.append(ans)
                if ans == YES_TOKEN:
                    q_label = correct_label
                else:
                    q_label = random.choice(list(class_labels - {correct_label}))
                questions.append(
                    "# " + random.choice(self.question_paraphrases[q_label]).format(q_label)
                )

            context_qa.append(
                ContextQASample(
                    context=str(example["prompt"]).strip(),
                    questions=questions,
                    answers=answers,
                    ds_label=correct_label,
                )
            )

        return context_qa


class EngelsNewsClassPoliticsLoader(BaseEngelsCsvBinaryLoader):
    GROUP_NAME = "engels_news_class_politics"
    DATASET_NAME = "news_class_politics"
    FILE_NAME = "139_news_class_Politics.csv"
    LABELS = ("politics", "not_politics")


class EngelsHeadlineIsTrumpLoader(BaseEngelsCsvBinaryLoader):
    GROUP_NAME = "engels_headline_istrump"
    DATASET_NAME = "headline_istrump"
    FILE_NAME = "21_headline_istrump.csv"
    LABELS = ("trump", "not_trump")


class EngelsHeadlineIsObamaLoader(BaseEngelsCsvBinaryLoader):
    GROUP_NAME = "engels_headline_isobama"
    DATASET_NAME = "headline_isobama"
    FILE_NAME = "22_headline_isobama.csv"
    LABELS = ("obama", "not_obama")


class EngelsHeadlineIsChinaLoader(BaseEngelsCsvBinaryLoader):
    GROUP_NAME = "engels_headline_ischina"
    DATASET_NAME = "headline_ischina"
    FILE_NAME = "23_headline_ischina.csv"
    LABELS = ("china", "not_china")


class EngelsHistFigIsMaleLoader(BaseEngelsCsvBinaryLoader):
    GROUP_NAME = "engels_hist_fig_ismale"
    DATASET_NAME = "hist_fig_ismale"
    FILE_NAME = "5_hist_fig_ismale.csv"
    LABELS = ("male", "female")


class EngelsWikidataIsJournalistLoader(BaseEngelsCsvBinaryLoader):
    GROUP_NAME = "engels_wikidata_isjournalist"
    DATASET_NAME = "wikidata_isjournalist"
    FILE_NAME = "59_wikidata_occupation_isjournalist.csv"
    LABELS = ("journalist", "not_journalist")


class EngelsWikidataIsAthleteLoader(BaseEngelsCsvBinaryLoader):
    GROUP_NAME = "engels_wikidata_isathlete"
    DATASET_NAME = "wikidata_isathlete"
    FILE_NAME = "60_wikidata_occupation_isathlete.csv"
    LABELS = ("athlete", "not_athlete")


class EngelsWikidataIsPoliticianLoader(BaseEngelsCsvBinaryLoader):
    GROUP_NAME = "engels_wikidata_ispolitician"
    DATASET_NAME = "wikidata_ispolitician"
    FILE_NAME = "62_wikidata_occupation_ispolitician.csv"
    LABELS = ("politician", "not_politician")


class EngelsWikidataIsSingerLoader(BaseEngelsCsvBinaryLoader):
    GROUP_NAME = "engels_wikidata_issinger"
    DATASET_NAME = "wikidata_issinger"
    FILE_NAME = "63_wikidata_occupation_issinger.csv"
    LABELS = ("singer", "not_singer")


class EngelsWikidataIsResearcherLoader(BaseEngelsCsvBinaryLoader):
    GROUP_NAME = "engels_wikidata_isresearcher"
    DATASET_NAME = "wikidata_isresearcher"
    FILE_NAME = "64_wikidata_occupation_isresearcher.csv"
    LABELS = ("researcher", "not_researcher")

class LanguageIDDatasetLoader(DatasetLoader):
    GROUP_NAME = "language_identification"
    DATASET_NAME = "language_identification"

    def __init__(self):
        super().__init__(self.__class__.GROUP_NAME, self.__class__.DATASET_NAME)

    def load(self, num_qa_per_sample: int) -> list[ContextQASample]:
        hf_dataset = load_dataset("FrancophonIA/WiLI-2018")["train"]
        context_qa: list[ContextQASample] = []
        class_labels = set(hf_dataset[:]["language"])

        for example in hf_dataset:
            questions = []
            answers = []
            correct_label = example["language"]
            for idx in range(num_qa_per_sample):
                ans = random.choice([YES_TOKEN, NO_TOKEN])
                ques = "# " + random.choice(self.question_paraphrases)
                answers.append(ans)
                if ans == YES_TOKEN:
                    questions.append(ques.format(correct_label))
                else:
                    incorrect_label = random.choice(list(class_labels - {correct_label}))
                    questions.append(ques.format(incorrect_label))

            context_qa.append(
                ContextQASample(
                    context=example["Text"],
                    questions=questions,
                    answers=answers,
                    ds_label=correct_label,
                )
            )

        return context_qa


class DatasetManager:
    supported_datasets: dict[tuple[str, str], DatasetLoader] = {
        (dataset.group, dataset.name): dataset
        for dataset in (
            GeometryOfTruthDatasetLoader.get_all_loaders()
            + RelationDatasetLoader.get_all_loaders()
            + EngelsDatasetLoader.get_all_loaders()
            + [
                EngelsNewsClassPoliticsLoader(),
                EngelsHeadlineIsTrumpLoader(),
                EngelsHeadlineIsObamaLoader(),
                EngelsHeadlineIsChinaLoader(),
                EngelsHistFigIsMaleLoader(),
                EngelsWikidataIsJournalistLoader(),
                EngelsWikidataIsAthleteLoader(),
                EngelsWikidataIsPoliticianLoader(),
                EngelsWikidataIsSingerLoader(),
                EngelsWikidataIsResearcherLoader(),
            ]
            + [
                SstDatasetLoader(),
                MdGenderDatasetLoader(),
                SnliDatasetLoader(),
                AgNewsDatasetLoader(),
                NerDatasetLoader(),
                TenseDatasetLoader(),
                LanguageIDDatasetLoader(),
                SingularPluralDatasetLoader(),
            ]
        )
    }

    def __init__(self, examples, batch_size, shuffle):
        self.examples = examples
        self.batch_size = batch_size

        if shuffle:
            random.shuffle(self.examples)

    def split(self, proportions):
        assert sum(proportions) <= 1

        start = 0
        end = None
        result = []
        for proportion in proportions:
            end = start + math.ceil(proportion * len(self.examples))
            result.append(DatasetManager(self.examples[start:end], self.batch_size, shuffle=False))
            start = end
        return result

    def __len__(self):
        return (len(self.examples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.examples), self.batch_size):
            yield self.examples[i : i + self.batch_size]

    @staticmethod
    def from_named_datasets(dataset_names, batch_size=1, shuffle=True):
        examples = []
        for group, name in dataset_names:
            dataset = DatasetManager.supported_datasets[(group, name)]
            examples.extend(dataset.load())
        return DatasetManager(examples, batch_size, shuffle)

    @staticmethod
    def from_dataset_group(group, **kwargs):
        datasets = DatasetManager.list_datasets_by_group(group)
        names = datasets[group]

        return DatasetManager.from_named_datasets(zip([group] * len(names), names), **kwargs)

    @staticmethod
    def list_datasets_by_group(want_group=None):
        result = {}
        for group, name in DatasetManager.supported_datasets:
            if want_group is not None and group != want_group:
                continue
            if group not in result:
                result[group] = []
            result[group].append(name)
        return result


def get_samples_from_groups(group_names: list[str], num_qa_per_sample: int) -> list[ContextQASample]:
    """
    Get all ContextQASample objects from specified groups.

    Args:
        group_names: List of group names (e.g., ["sst2", "ag_news"])
        num_qa_per_sample: Number of Q&A pairs per sample (default 10)

    Returns:
        List of ContextQASample objects
    """
    all_samples = []

    for group in group_names:
        # Get all dataset names for this group
        datasets_in_group = DatasetManager.list_datasets_by_group(group)[group]
        assert len(datasets_in_group) > 0, f"No datasets found for group {group}"

        for dataset_name in datasets_in_group:
            # Get the loader
            loader = DatasetManager.supported_datasets[(group, dataset_name)]
            assert loader is not None, f"Loader not found for dataset {dataset_name}"
            # Load the samples
            samples = loader.load(num_qa_per_sample)
            all_samples.extend(samples)

    for i in range(len(all_samples)):
        for j in range(len(all_samples[i].questions)):
            if all_samples[i].questions[j][:2] == "# ":
                all_samples[i].questions[j] = all_samples[i].questions[j][2:]

    return all_samples
