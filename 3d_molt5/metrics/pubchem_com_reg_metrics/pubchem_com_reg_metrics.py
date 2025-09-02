import datasets
import evaluate
import re
import numpy as np
from sklearn import metrics

_CITATION = """\
@article{xxx
}
"""

_DESCRIPTION = """\
Classification metrics for DTI.
"""

_KWARGS_DESCRIPTION = """
No args.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class pubchem_com_regression(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://xxx.com"],
            reference_urls=[
                "https://xxx.com"
            ],
        )

    def _compute(self, predictions, references, tsv_path="tmp.tsv"):
        # 'Topological Polar Surface Area', 'Molecular Weight', 'LogP', 'Complexity'
        predictions_weight = []
        references_weight = []

        predictions_logp = []
        references_logp = []

        predictions_tpsa = []
        references_tpsa = []

        predictions_complexity = []
        references_complexity = []

        counter_weight, counter_logp, counter_tpsa, counter_complexity = 0, 0, 0, 0

        pattern = r'-?\d+\.\d+'

        for pred, ref in zip(predictions, references):
            pred_matches = re.findall(pattern, pred)
            gt_matches = re.findall(pattern, ref)
            if len(pred_matches) > 0:
                if 'Molecular Weight' in ref and 0 < float(pred_matches[0]) < 4000:
                    predictions_weight.append(float(pred_matches[0]))
                    references_weight.append(float(gt_matches[0]))
                elif 'LogP' in ref and -30 < float(pred_matches[0]) < 50:
                    predictions_logp.append(float(pred_matches[0]))
                    references_logp.append(float(gt_matches[0]))
                elif 'Topological Polar Surface Area' in ref and 0 <= float(pred_matches[0]) < 2000:
                    predictions_tpsa.append(float(pred_matches[0]))
                    references_tpsa.append(float(gt_matches[0]))
                elif 'Complexity' in ref and 0 <= float(pred_matches[0]) < 10000:
                    predictions_complexity.append(float(pred_matches[0]))
                    references_complexity.append(float(gt_matches[0]))
            if 'Molecular Weight' in ref:
                counter_weight += 1
            elif 'LogP' in ref:
                counter_logp += 1
            elif 'Topological Polar Surface Area' in ref:
                counter_tpsa += 1
            elif 'Complexity' in ref:
                counter_complexity += 1
        # Function to calculate RMSE and MAE, returns None if lists are empty
        def calc_metrics(refs, preds):
            if len(refs) > 0 and len(preds) > 0:
                rmse = np.sqrt(metrics.mean_squared_error(refs, preds))
                mae = metrics.mean_absolute_error(refs, preds)
            else:
                rmse, mae = 0.0, 0.0
            return rmse, mae
        # Calculate metrics: RMSE, MAE
        rmse_weight, mae_weight = calc_metrics(references_weight, predictions_weight)
        rmse_logp, mae_logp = calc_metrics(references_logp, predictions_logp)
        rmse_tpsa, mae_tpsa = calc_metrics(references_tpsa, predictions_tpsa)
        rmse_complexity, mae_complexity = calc_metrics(references_complexity, predictions_complexity)

        return {
            "mae_weight": mae_weight,
            "valid_weight": len(predictions_weight) / counter_weight,
            "mae_logp": mae_logp,
            "valid_logp": len(predictions_logp) / counter_logp,
            "mae_tpsa": mae_tpsa,
            "valid_tpsa": len(predictions_tpsa) / counter_tpsa,
            "mae_complexity": mae_complexity,
            "valid_complexity": len(predictions_complexity) / counter_complexity,
        }
