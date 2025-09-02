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
class pqc_prop_regression(evaluate.Metric):
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

        predictions_homo = []
        references_homo = []

        predictions_lumo = []
        references_lumo = []

        predictions_gap = []
        references_gap = []

        predictions_scf = []
        references_scf = []

        counter_scf, counter_homo, counter_lumo, counter_gap = 0, 0, 0, 0

        pattern = r'-?\d+\.\d+'

        for pred, ref in zip(predictions, references):
            pred_matches = re.findall(pattern, pred)
            gt_matches = re.findall(pattern, ref)
            if len(pred_matches) > 0:
                if 'SCF Energy' in ref and -5 < float(pred_matches[0]) < 0:
                    predictions_scf.append(float(pred_matches[0]))
                    references_scf.append(float(gt_matches[0]))
                elif 'HOMO-LUMO Gap' in ref and -20 < float(pred_matches[0]) < 20:
                    predictions_gap.append(float(pred_matches[0]))
                    references_gap.append(float(gt_matches[0]))
                elif 'HOMO' in ref and 'LUMO' not in ref and -20 < float(pred_matches[0]) < 20:
                    predictions_homo.append(float(pred_matches[0]))
                    references_homo.append(float(gt_matches[0]))
                elif 'LUMO' in ref and 'HOMO' not in ref and -20 < float(pred_matches[0]) < 20:
                    predictions_lumo.append(float(pred_matches[0]))
                    references_lumo.append(float(gt_matches[0]))
            if 'SCF Energy' in ref:
                counter_scf += 1
            elif 'HOMO-LUMO Gap' in ref:
                counter_gap += 1
            elif 'HOMO' in ref and 'LUMO' not in ref:
                counter_homo += 1
            elif 'LUMO' in ref and 'HOMO' not in ref:
                counter_lumo += 1
        # Function to calculate RMSE and MAE, returns None if lists are empty
        def calc_metrics(refs, preds):
            if len(refs) > 0 and len(preds) > 0:
                rmse = np.sqrt(metrics.mean_squared_error(refs, preds))
                mae = metrics.mean_absolute_error(refs, preds)
            else:
                rmse, mae = 0.0, 0.0
            return rmse, mae
        # Calculate metrics: RMSE, MAE
        rmse_scf, mae_scf = calc_metrics(references_scf, predictions_scf)
        rmse_gap, mae_gap = calc_metrics(references_gap, predictions_gap)
        rmse_homo, mae_homo = calc_metrics(references_homo, predictions_homo)
        rmse_lumo, mae_lumo = calc_metrics(references_lumo, predictions_lumo)

        return {
            "mae_homo": mae_homo,
            "valid_homo": len(predictions_homo) / counter_homo if len(predictions_homo) > 0 else 0,
            "mae_lumo": mae_lumo,
            "valid_lumo": len(predictions_lumo) / counter_lumo if len(predictions_lumo) > 0 else 0,
            "mae_gap": mae_gap,
            "valid_gap": len(predictions_gap) / counter_gap if len(predictions_gap) > 0 else 0,
            "mae_scf": mae_scf,
            "valid_scf": len(predictions_scf) / counter_scf if len(predictions_scf) > 0 else 0,
        }