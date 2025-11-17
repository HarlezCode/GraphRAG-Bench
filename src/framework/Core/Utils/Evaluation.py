"""
Evaluation utilities for GraphRAG system performance assessment.
Provides comprehensive evaluation metrics for different types of RAG systems and datasets.
"""

import json
import pandas as pd
import re
import string
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple, Union
import copy
import numpy as np
import mauve
import nltk
from nltk import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer, scoring

# Import default_config as type hint only to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Option.merged_config import default_config
# Import BaseLLM as type hint only to avoid circular import
if TYPE_CHECKING:
    from Core.Provider.BaseLLM import BaseLLM
# Import create_llm_instance as type hint only to avoid circular import
if TYPE_CHECKING:
    from Core.Provider.LLMProviderRegister import create_llm_instance

# NLTK setup
nltk_path = "YOUR_OWN_NLTK_CACHE_PATH"
nltk.data.path.append(nltk_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading punkt")
    nltk.download("punkt", download_dir=nltk_path)
    nltk.download("wordnet", download_dir=nltk_path)


class MetricCalculator:
    """
    Calculator for various evaluation metrics.
    
    Provides methods to calculate BLEU, METEOR, ROUGE, and other
    text similarity metrics used in RAG evaluation.
    """
    
    def __init__(self):
        """Initialize the metric calculator."""
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    
    @staticmethod
    def calculate_bleu_1(prediction: str, references: List[str]) -> float:
        """
        Calculate BLEU-1 score.
        
        Args:
            prediction: Predicted text
            references: List of reference texts
            
        Returns:
            BLEU-1 score
        """
        return sentence_bleu(references, prediction, weights=(1, 0, 0, 0))
    
    @staticmethod
    def calculate_bleu_4(prediction: str, references: List[str]) -> float:
        """
        Calculate BLEU-4 score.
        
        Args:
            prediction: Predicted text
            references: List of reference texts
            
        Returns:
            BLEU-4 score
        """
        return sentence_bleu(references, prediction, weights=(0, 0, 0, 1))
    
    @staticmethod
    def calculate_bleu_4_modified(prediction: str, references: List[str]) -> float:
        """
        Calculate modified BLEU-4 score with equal weights.
        
        Args:
            prediction: Predicted text
            references: List of reference texts
            
        Returns:
            Modified BLEU-4 score
        """
        return sentence_bleu(references, prediction, weights=(0.25, 0.25, 0.25, 0.25))
    
    @staticmethod
    def calculate_bleu_1_smooth(prediction: str, references: List[str]) -> float:
        """
        Calculate BLEU-1 score with smoothing.
        
        Args:
            prediction: Predicted text
            references: List of reference texts
            
        Returns:
            Smoothed BLEU-1 score
        """
        return sentence_bleu(
            references, prediction, weights=(1, 0, 0, 0), 
            smoothing_function=SmoothingFunction().method1
        )
    
    @staticmethod
    def calculate_bleu_4_smooth(prediction: str, references: List[str]) -> float:
        """
        Calculate BLEU-4 score with smoothing.
        
        Args:
            prediction: Predicted text
            references: List of reference texts
            
        Returns:
            Smoothed BLEU-4 score
        """
        return sentence_bleu(
            references, prediction, weights=(0, 0, 0, 1), 
            smoothing_function=SmoothingFunction().method1
        )
    
    @staticmethod
    def calculate_bleu_4_modified_smooth(prediction: str, references: List[str]) -> float:
        """
        Calculate modified BLEU-4 score with smoothing.
        
        Args:
            prediction: Predicted text
            references: List of reference texts
            
        Returns:
            Smoothed modified BLEU-4 score
        """
        return sentence_bleu(
            references, prediction, weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1
        )
    
    def calculate_meteor(self, prediction: str, references: List[str]) -> float:
        """
        Calculate METEOR score.
        
        Args:
            prediction: Predicted text
            references: List of reference texts
            
        Returns:
            METEOR score
        """
        return meteor_score([ref.split() for ref in references], prediction.split())
    
    def calculate_rouge_l(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE-L score.
        
        Args:
            prediction: Predicted text
            references: List of reference texts
            
        Returns:
            Dictionary containing ROUGE-L precision, recall, and F1
        """
        if isinstance(references, list):
            reference = references[0]
        else:
            reference = references
        
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            "rouge_l_f1": scores["rougeL"].fmeasure,
            "rouge_l_precision": scores["rougeL"].precision,
            "rouge_l_recall": scores["rougeL"].recall
        }


class TextNormalizer:
    """
    Text normalization utilities for evaluation.
    
    Provides methods to normalize text for consistent evaluation
    across different formats and styles.
    """
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """
        Normalize text for evaluation by removing articles, punctuation, etc.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)
        
        def white_space_fix(text: str) -> str:
            return " ".join(text.split())
        
        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        
        def lower(text: str) -> str:
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(text))))
    
    @staticmethod
    def normalize_prediction(prediction: str) -> str:
        """
        Normalize prediction text by handling special separators.
        
        Args:
            prediction: Prediction text to normalize
            
        Returns:
            Normalized prediction text
        """
        prediction = prediction.replace("|", "\n")
        prediction = prediction.split("\n")
        return " ".join(prediction)


class EvaluationMetrics:
    """
    Collection of evaluation metrics for different evaluation modes.
    """
    
    SHORT_FORM_METRICS = ["accuracy", "f1", "precision", "recall", "em"]
    CLOSE_SET_METRICS = ["accuracy"]
    LONG_NARRATIVE_METRICS = [
        "bleu_1", "bleu_4", "modify_bleu_4", "bleu_1_smooth", 
        "bleu_4_smooth", "modify_bleu_4_smooth", "meteor",
        "rouge_l_f1", "rouge_l_precision", "rouge_l_recall"
    ]
    LONG_ASQA_METRICS = ["str_em", "str_hit", "rougeLsum", "mauve"]


class DatasetModeMapper:
    """
    Maps dataset names to their evaluation modes.
    """
    
    DATASET_MODE_MAP = {
        "hotpotqa": "short-form",
        "multihop-rag": "short-form",
        "popqa": "short-form",
        "ALCE": "long-asqa",
        "quality": "close-set",
    }
    
    @classmethod
    def get_mode(cls, dataset_name: str) -> str:
        """
        Get evaluation mode for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Evaluation mode
        """
        if "narrative" in dataset_name:
            return "long-narrative"
        return cls.DATASET_MODE_MAP.get(dataset_name, "short-form")


class MetricAggregator:
    """
    Aggregates metrics across multiple predictions and references.
    """
    
    @staticmethod
    def metric_max_over_ground_truths(
        metric_fn, 
        prediction: str, 
        ground_truths: List[str], 
        tokenize: bool = False
    ) -> Union[float, Dict[str, float]]:
        """
        Calculate maximum metric score over multiple ground truths.
        
        Args:
            metric_fn: Metric function to apply
            prediction: Prediction text
            ground_truths: List of ground truth texts
            tokenize: Whether to tokenize text
            
        Returns:
            Maximum metric score or score dictionary
        """
        scores_for_ground_truths = []
        
        for ground_truth in ground_truths:
            if tokenize:
                score = metric_fn(word_tokenize(prediction), [word_tokenize(ground_truth)])
            else:
                score = metric_fn(prediction, [ground_truth])
            scores_for_ground_truths.append(score)
        
        if isinstance(score, dict) and "rougeL" in score:
            rouge_l_score = {
                "rouge_l_f1": 0, 
                "rouge_l_precision": 0, 
                "rouge_l_recall": 0
            }
            rouge_l_score["rouge_l_f1"] = max(
                [score["rougeL"].fmeasure for score in scores_for_ground_truths]
            )
            rouge_l_score["rouge_l_precision"] = max(
                [score["rougeL"].precision for score in scores_for_ground_truths]
            )
            rouge_l_score["rouge_l_recall"] = max(
                [score["rougeL"].recall for score in scores_for_ground_truths]
            )
            return rouge_l_score
        else:
            return round(max(scores_for_ground_truths), 2)
    
    @staticmethod
    def get_all_metric_scores(prediction: str, ground_truths: List[str]) -> Dict[str, float]:
        """
        Calculate all available metrics for a prediction.
        
        Args:
            prediction: Prediction text
            ground_truths: List of ground truth texts
            
        Returns:
            Dictionary of all metric scores
        """
        calculator = MetricCalculator()
        
        bleu_1_score = MetricAggregator.metric_max_over_ground_truths(
            calculator.calculate_bleu_1, prediction, ground_truths, tokenize=True
        )
        bleu_4_score = MetricAggregator.metric_max_over_ground_truths(
            calculator.calculate_bleu_4, prediction, ground_truths, tokenize=True
        )
        modify_bleu_4_score = MetricAggregator.metric_max_over_ground_truths(
            calculator.calculate_bleu_4_modified, prediction, ground_truths, tokenize=True
        )
        bleu_1_smooth_score = MetricAggregator.metric_max_over_ground_truths(
            calculator.calculate_bleu_1_smooth, prediction, ground_truths, tokenize=True
        )
        bleu_4_smooth_score = MetricAggregator.metric_max_over_ground_truths(
            calculator.calculate_bleu_4_smooth, prediction, ground_truths, tokenize=True
        )
        modify_bleu_4_smooth_score = MetricAggregator.metric_max_over_ground_truths(
            calculator.calculate_bleu_4_modified_smooth, prediction, ground_truths, tokenize=True
        )
        meteor_score = MetricAggregator.metric_max_over_ground_truths(
            calculator.calculate_meteor, prediction, ground_truths, tokenize=False
        )
        rouge_l_score = MetricAggregator.metric_max_over_ground_truths(
            calculator.calculate_rouge_l, prediction, ground_truths, tokenize=False
        )
        
        return {
            "bleu_1": bleu_1_score,
            "bleu_4": bleu_4_score,
            "modify_bleu_4": modify_bleu_4_score,
            "bleu_1_smooth": bleu_1_smooth_score,
            "bleu_4_smooth": bleu_4_smooth_score,
            "modify_bleu_4_smooth": modify_bleu_4_smooth_score,
            "meteor": meteor_score,
            "rouge_l_f1": rouge_l_score["rouge_l_f1"],
            "rouge_l_precision": rouge_l_score["rouge_l_precision"],
            "rouge_l_recall": rouge_l_score["rouge_l_recall"],
        }


class Evaluator:
    """
    Main evaluator class for GraphRAG system performance assessment.
    
    Provides comprehensive evaluation capabilities for different types of
    RAG systems and datasets with support for multiple evaluation modes.
    """
    
    def __init__(self, eval_path: str, dataset_name: str):
        """
        Initialize evaluator.
        
        Args:
            eval_path: Path to evaluation results file
            dataset_name: Name of the dataset
        """
        self.eval_path = eval_path
        self.dataset_name = dataset_name
        from Option.merged_config import default_config
        self.config = default_config
        from Core.Provider.LLMProviderRegister import create_llm_instance
        self.llm = create_llm_instance(self.config.llm)
        self.metric_calculator = MetricCalculator()
        self.text_normalizer = TextNormalizer()
        self.mode = DatasetModeMapper.get_mode(dataset_name)
    
    async def evaluate(self) -> Dict[str, float]:
        """
        Perform evaluation on the dataset.
        
        Returns:
            Dictionary containing evaluation results
        """
        import ipdb; ipdb.set_trace()
        
        df = pd.read_json(self.eval_path, lines=True)
        print(f"Loaded {len(df)} records from {self.eval_path}")
        print(f"Evaluating {self.mode} mode.")
        
        # Select evaluation method based on mode
        if self.mode == "short-form":
            self._print_eval_metrics(EvaluationMetrics.SHORT_FORM_METRICS)
            res_dict, df = self._short_form_eval(df)
        elif self.mode == "long-narrative":
            self._print_eval_metrics(EvaluationMetrics.LONG_NARRATIVE_METRICS)
            res_dict, df = self._long_narrative_eval(df)
        elif self.mode == "long-asqa":
            self._print_eval_metrics(EvaluationMetrics.LONG_ASQA_METRICS)
            res_dict, df = self._long_asqa_eval(df)
        elif self.mode == "close-set":
            self._print_eval_metrics(EvaluationMetrics.CLOSE_SET_METRICS)
            res_dict, df = await self._close_set_eval(df)
        else:
            raise ValueError(f"Invalid evaluation mode: {self.mode}")
        
        # Save results
        save_path = self.eval_path.replace(".json", ".score.json")
        df.to_json(save_path, orient="records", lines=True)
        return res_dict
    
    def _print_eval_metrics(self, eval_metrics: List[str]) -> None:
        """
        Print the metrics being used for evaluation.
        
        Args:
            eval_metrics: List of metric names
        """
        print("In this evaluation, the following metrics are used:")
        for metric in eval_metrics:
            print(metric, end=" ")
        print("\n")
    
    def _get_label_pred_list(self, df: pd.DataFrame, pred_col: str, label_col: str) -> Tuple[List[str], List[str]]:
        """
        Extract prediction and label lists from dataframe.
        
        Args:
            df: Input dataframe
            pred_col: Prediction column name
            label_col: Label column name
            
        Returns:
            Tuple of (label_list, pred_list)
        """
        label_list = df[label_col].tolist()
        pred_list = df[pred_col].tolist()
        return label_list, pred_list
    
    def _short_form_eval(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Perform short-form evaluation (accuracy, F1, precision, recall, EM).
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (results_dict, updated_dataframe)
        """
        # Short form evaluation code is referenced from the HippoRAG evaluation script:
        # links: https://github.com/OSU-NLP-Group/HippoRAG
        
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        em_list = []
        
        label_list, pred_list = self._get_label_pred_list(df, "output", "answer")
        
        for prediction, answer in zip(pred_list, label_list):
            prediction_str = self.text_normalizer.normalize_prediction(prediction)
            
            # Handle answer format
            if isinstance(answer, list):
                answer_str = " ".join(answer)
            else:
                answer_str = answer
            
            # Calculate metrics
            accuracy = self._eval_accuracy(prediction_str, answer_str)
            f1, precision, recall = self._f1_score(prediction_str, answer_str)
            em = self._exact_match_score(prediction_str, answer_str)
            
            # Store results
            em_list.append(em)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
        
        # Calculate averages
        accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        f1 = sum(f1_list) * 100 / len(f1_list)
        precision = sum(precision_list) * 100 / len(precision_list)
        recall = sum(recall_list) * 100 / len(recall_list)
        em = sum(em_list) * 100 / len(em_list)
        
        # Update dataframe
        df["accuracy"] = accuracy_list
        df["f1"] = f1_list
        df["precision"] = precision_list
        df["recall"] = recall_list
        df["em"] = em_list
        
        res_dict = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "em": em,
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"EM: {em:.4f}")
        
        return res_dict, df
    
    def _long_narrative_eval(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Perform long narrative evaluation (BLEU, METEOR, ROUGE).
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (results_dict, updated_dataframe)
        """
        label_list, pred_list = self._get_label_pred_list(df, "output", "answer")
        
        # Initialize metric lists
        metric_lists = {
            "bleu_1": [], "bleu_4": [], "modify_bleu_4": [],
            "bleu_1_smooth": [], "bleu_4_smooth": [], "modify_bleu_4_smooth": [],
            "meteor": [], "rouge_l_f1": [], "rouge_l_precision": [], "rouge_l_recall": []
        }
        
        for prediction, answer in zip(pred_list, label_list):
            prediction_str = self.text_normalizer.normalize_prediction(prediction)
            
            # Calculate all metrics
            metrics_res = MetricAggregator.get_all_metric_scores(prediction_str, answer)
            
            # Store results
            for metric_name in metric_lists:
                metric_lists[metric_name].append(metrics_res[metric_name])
        
        # Calculate averages
        results = {}
        for metric_name, values in metric_lists.items():
            avg_value = sum(values) * 100 / len(values)
            results[metric_name] = avg_value
            df[metric_name] = values
            print(f"{metric_name.replace('_', ' ').title()}: {avg_value:.4f}")
        
        return results, df
    
    def _long_asqa_eval(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Perform long ASQA evaluation (str_em, str_hit, ROUGE-Lsum, MAUVE).
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (results_dict, updated_dataframe)
        """
        # Long ASQA code is referenced from official ALCE repository:
        # links: https://github.com/princeton-nlp/ALCE
        
        str_em_list = []
        str_hit_list = []
        
        for index, row in df.iterrows():
            prediction = row["output"]
            answer_pairs = row["qa_pairs"]
            
            prediction_str = self.text_normalizer.normalize_prediction(prediction)
            prediction_str = self.text_normalizer.normalize_answer(prediction_str)
            
            str_em, str_hit = self._eval_str_em(prediction_str, answer_pairs)
            str_em_list.append(str_em)
            str_hit_list.append(str_hit)
        
        # Calculate MAUVE and ROUGE-Lsum
        mauve_score = self._compute_mauve(df)
        rouge_lsum_score = self._compute_rouge(df)
        
        # Calculate averages
        str_em = sum(str_em_list) * 100 / len(str_em_list)
        str_hit = sum(str_hit_list) * 100 / len(str_hit_list)
        
        # Update dataframe
        df["str_em"] = str_em_list
        df["str_hit"] = str_hit_list
        df["rougeLsum"] = rouge_lsum_score
        
        res_dict = {
            "str_em": str_em,
            "str_hit": str_hit,
            "mauve": mauve_score,
            "rougeLsum": rouge_lsum_score,
        }
        
        # Print results
        print(f"str_em: {str_em:.4f}")
        print(f"str_hit: {str_hit:.4f}")
        print(f"mauve: {mauve_score:.4f}")
        print(f"rougeLsum: {rouge_lsum_score:.4f}")
        
        return res_dict, df
    
    async def _close_set_eval(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Perform close-set evaluation using LLM to extract options.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (results_dict, updated_dataframe)
        """
        # Close set evaluation first uses LLM to extract the option from the model output
        # Then evaluates the extracted option with the answer index
        
        for index, row in df.iterrows():
            prompt = CLOSE_EXTRACT_OPTION_PROMPT.format(
                question=row["question"], model_output=row["output"]
            )
            response = await self.llm.aask(msg=prompt, format="json")
            
            try:
                df.loc[index, "extract_output"] = response["predict"]
            except Exception as e:
                df.loc[index, "extract_output"] = "-1"
        
        print("LLM extract option completed.")
        
        # Calculate accuracy
        accuracy_list = []
        label_list, pred_list = self._get_label_pred_list(df, "extract_output", "answer_idx")
        
        for prediction, answer in zip(pred_list, label_list):
            prediction = prediction.strip()
            answer = answer.strip()
            accuracy = self._exact_match_score(prediction, answer)
            accuracy_list.append(accuracy)
        
        accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        df["accuracy"] = accuracy_list
        
        res_dict = {"accuracy": accuracy}
        print(f"Accuracy: {accuracy:.4f}")
        
        return res_dict, df
    
    def _exact_presence(self, short_answers: List[str], context: str) -> bool:
        """
        Verify if any of the answers is present in the given context.
        
        Args:
            short_answers: List of short answers to look for
            context: Context to search in
            
        Returns:
            True if any answer is found in context
        """
        normalized_answers = [self.text_normalizer.normalize_answer(sa) for sa in short_answers]
        normalized_context = self.text_normalizer.normalize_answer(context)
        
        for answer in normalized_answers:
            if answer in normalized_context:
                return True
        
        return False
    
    def _eval_str_em(self, prediction: str, qa_pairs: List[Dict]) -> Tuple[float, int]:
        """
        Evaluate string exact match for QA pairs.
        
        Args:
            prediction: Prediction text
            qa_pairs: List of QA pairs
            
        Returns:
            Tuple of (accuracy, hit)
        """
        if len(qa_pairs) == 0:
            return 0, 0
        
        loc_acc = []
        for qa_pair in qa_pairs:
            loc_acc.append(self._exact_presence(qa_pair["short_answers"], prediction))
        
        acc = np.mean(loc_acc)
        hit = int(acc == 1)
        
        return acc, hit
    
    def _compute_mauve(self, df: pd.DataFrame) -> float:
        """
        Compute MAUVE score for the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            MAUVE score
        """
        human_data = []
        model_data = []
        
        for idx, row in df.iterrows():
            # Remove ending punctuations, new lines, and truncate by 100 words
            human_data.append(
                " ".join(
                    (row["question"] + " " + row["answer"].strip()).split()[:100]
                ).rstrip(string.punctuation)
            )
            model_data.append(
                " ".join(
                    (row["question"] + " " + row["output"].strip()).split()[:100]
                ).rstrip(string.punctuation)
            )
        
        out = mauve.compute_mauve(
            p_text=human_data,
            q_text=model_data,
            device_id=0,
            max_text_length=512,
            verbose=True,
            batch_size=8,
            featurize_model_name="gpt2-large",
        )
        return out.mauve * 100
    
    def _compute_rouge(self, df: pd.DataFrame) -> float:
        """
        Compute ROUGE-Lsum score for the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            ROUGE-Lsum score
        """
        def _rouge_calculation(
            hypotheses: List[str], 
            references1: List[str], 
            references2: List[str] = None, 
            metrics: List[str] = ["rougeLsum"]
        ) -> Dict[str, float]:
            if references2 is None:
                references2 = references1
            
            scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
            aggregator = scoring.BootstrapAggregator()
            
            for i in range(len(hypotheses)):
                scores1 = scorer.score(references1[i], hypotheses[i])
                scores2 = scorer.score(references2[i], hypotheses[i])
                
                if scores1["rougeLsum"].fmeasure > scores2["rougeLsum"].fmeasure:
                    aggregator.add_scores(scores1)
                else:
                    aggregator.add_scores(scores2)
            
            scores = {m: [] for m in metrics}
            
            for m in metrics:
                fmeasure = aggregator.aggregate()[m].mid.fmeasure
                scores[m].append(fmeasure)
            
            for m in scores:
                scores[m] = 100 * sum(scores[m]) / len(scores[m])
            
            return scores
        
        # Prepare data
        hypotheses = {}
        references1 = {}
        references2 = {}
        
        for idx, item in df.iterrows():
            hypotheses[idx] = item["output"]
            if "annotations" in item and item["annotations"] is not None:  # For ASQA
                references1[idx] = item["annotations"][0]["long_answer"]
                references2[idx] = item["annotations"][1]["long_answer"]
            else:
                references1[idx] = item["answer"]
                references2[idx] = item["answer"]
        
        h, r1, r2 = [], [], []
        
        for key in references1:
            h.append(hypotheses[key])
            r1.append(references1[key])
            if references2 is not None:
                r2.append(references2[key])
        
        # Process text
        h = ["\n".join(sent_tokenize(text.lower())) for text in h]
        r1 = ["\n".join(sent_tokenize(text.lower())) for text in r1]
        r2 = ["\n".join(sent_tokenize(text.lower())) for text in r2]
        
        scores = _rouge_calculation(h, r1, r2)
        return scores["rougeLsum"]
    
    def _f1_score(self, prediction: str, ground_truth: str) -> Tuple[float, float, float]:
        """
        Calculate F1 score between prediction and ground truth.
        
        Args:
            prediction: Prediction text
            ground_truth: Ground truth text
            
        Returns:
            Tuple of (f1, precision, recall)
        """
        normalized_prediction = self.text_normalizer.normalize_answer(prediction)
        normalized_ground_truth = self.text_normalizer.normalize_answer(ground_truth)
        
        ZERO_METRIC = (0, 0, 0)
        
        # Handle special cases
        if (normalized_prediction in ["yes", "no", "noanswer"] 
            and normalized_prediction != normalized_ground_truth):
            return ZERO_METRIC
        if (normalized_ground_truth in ["yes", "no", "noanswer"] 
            and normalized_prediction != normalized_ground_truth):
            return ZERO_METRIC
        
        # Calculate F1
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return ZERO_METRIC
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1, precision, recall
    
    def _exact_match_score(self, prediction: str, ground_truth: str) -> int:
        """
        Calculate exact match score.
        
        Args:
            prediction: Prediction text
            ground_truth: Ground truth text
            
        Returns:
            1 if exact match, 0 otherwise
        """
        return int(self.text_normalizer.normalize_answer(prediction) == 
                  self.text_normalizer.normalize_answer(ground_truth))
    
    def _eval_accuracy(self, prediction: str, ground_truth: str) -> int:
        """
        Calculate accuracy score (substring match).
        
        Args:
            prediction: Prediction text
            ground_truth: Ground truth text
            
        Returns:
            1 if ground truth is substring of prediction, 0 otherwise
        """
        normalized_prediction = self.text_normalizer.normalize_answer(prediction)
        normalized_ground_truth = self.text_normalizer.normalize_answer(ground_truth)
        
        return int(normalized_ground_truth in normalized_prediction)


# Prompt template for close-set evaluation
CLOSE_EXTRACT_OPTION_PROMPT = """
You are given a model output which is a string. The model output is a list of options. You have to extract the option letter from the model output.

# GOAL

Your goal is to extract the option letter directly from the model output. You should not rely on any external knowledge or context to answer. Simply extract the option letter as stated in the model output.

# FORMAT

Please provide your answer in the following JSON format:

- ANSWER_OPTION: the option letter extracted from the model output.

    {{
        "model_output": <answer_option>
    }}

### Example 1
-----------
# INPUT:

Question:
How much time has passed between Blake's night with Eldoria and his search for Sabrina York in his mind-world?
A: 7 years
B: 10 hours
C: 12 years
D: 1 hour

# Model Output: 
I think the answer is 7 years.

OUTPUT:
    {{
        "predict": "A"
    }}

### Example 2
-----------
# INPUT:

Question:
How much time has passed between Blake's night with Eldoria and his search for Sabrina York in his mind-world?
A: 7 years
B: 10 hours
C: 12 years
D: 1 hour

# Model Output: 
The correct answer is C.

OUTPUT:
    {{
        "predict": "C"
    }}
    
### EXAMPLE 3
-----------

# INPUT:

Question:
Donald Trump is the president of:
A: China
B: Canada
C: France
D: Spain

# Model Output: 
The correct answer is: None of the above.

OUTPUT:
    {{
        "predict": "-1"
    }}

Now please the output based on the given question and model output.

### Real Data
# INPUT:

Question:
{question}

# Model Output:
{model_output}

OUTPUT:"""
