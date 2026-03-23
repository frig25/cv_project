import os
import cv2
from tqdm import tqdm
from glob import glob
from sklearn.metrics import confusion_matrix
import pandas as pd


def accuracy(tn, fp, fn, tp):
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0

def precision(tn, fp, fn, tp):
    return tp / (tp + fp) if (tp + fp) > 0 else 1

def recall(tn, fp, fn, tp):
    return tp / (tp + fn) if (tp + fn) > 0 else 1

def f1_score(tn, fp, fn, tp):
    p = precision(tn, fp, fn, tp)
    r = recall(tn, fp, fn, tp)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 1

def iou(tn, fp, fn, tp):
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1

def create_html_with_images(alpha, gt_image_path, pred_image_path, output_path):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Images Side by Side</title>
        <style>
            .container {{
                display: flex;
                justify-content: center;
                align-items: flex-start;
                gap: 40px;
                margin-top: 50px;
                font-family: Arial, sans-serif;
            }}
            .image-block {{
                text-align: center;
            }}
            .label {{
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
                display: block;
                text-align: center;
            }}
            img {{
                max-width: 800px;
                height: auto;
                border: 2px solid #ccc;
            }}
        </style>
    </head>
    <body>
        <div class="label">ALPHA = {alpha}</div>
        <div class="container">
            <div class="image-block">
                <span class="label">GROUND TRUTH</span>
                <img src="{gt_image_path}" alt="Ground Truth">
            </div>
            <div class="image-block">
                <span class="label">PREDICTION</span>
                <img src="{pred_image_path}" alt="Prediction">
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)

def save_metrics_as_html(metrics_df, macro_df, micro_df, alpha, output_path):
    df_html = metrics_df.copy()

    df_html['img_id'] = df_html['img_id'].apply(
        lambda x: f'<a href="./comparison/{x}_comparison.html" target="_blank">{x}</a>'
    )

    num_rows = len(df_html)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Eval Metrics alpha {alpha}</title>
        <link rel="stylesheet" href="https://cdn.datatables.net/2.3.2/css/dataTables.dataTables.css" />
        <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.7.1.js"></script>
        <script src="https://cdn.datatables.net/2.3.2/js/dataTables.js"></script>
    </head>
    <body>
        <h2>EVALUATION METRICS - ALPHA = {alpha}</h2>

        MACRO OVERALL METRICS
        {macro_df.T.to_html()}

        MICRO OVERALL METRICS
        {micro_df.T.to_html()}

        {df_html.to_html(index=False, escape=False, classes="display", table_id="myTable")}
        <script>
            $(document).ready(function() {{
                $('#myTable').DataTable({{
                    "pageLength": {num_rows},
                    "paging": false
                }});
            }});
        </script>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    alpha=0.4

    # input
    gt_path = '../../data/udine/preprocessing/gt_mask'
    pred_path = f'../../data/udine/classification/masks/alpha_{alpha}/nhfd_pred_mask'
    gt_overlay_path = '../../../preprocessing/gt_overlay' # relative to comparison_path directory
    pred_overlay_path = f'../../../classification/masks/alpha_{alpha}/nhfd_overlay' # relative to comparison_path directory

    # output
    output_path = f'../../data/udine/evaluation/alpha_{alpha}'
    comparison_path = os.path.join(output_path, 'comparison')

    background_label = 0
    man_made_label = 255

    eval_metrics = {
        'img_id': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'iou': []
    }

    os.makedirs(comparison_path, exist_ok=True)

    gt_files = glob(os.path.join(gt_path, '*'))

    total_tn = total_fp = total_fn = total_tp = 0

    print('Computing evaluation metrics...')

    for gt_fp in tqdm(gt_files, total=len(gt_files)):
        img_id = os.path.basename(gt_fp)[:8]
        pred_fp = os.path.join(pred_path, f'{img_id}_pred_mask.png')
        if not os.path.exists(pred_fp):
            continue

        gt = cv2.imread(gt_fp, cv2.IMREAD_GRAYSCALE).flatten()
        pred = cv2.imread(pred_fp, cv2.IMREAD_GRAYSCALE).flatten()

        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[background_label, man_made_label]).ravel()

        # accumulate totals for micro metrics
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp

        # per-image metrics
        eval_metrics['img_id'].append(img_id)
        eval_metrics['accuracy'].append(accuracy(tn, fp, fn, tp))
        eval_metrics['precision'].append(precision(tn, fp, fn, tp))
        eval_metrics['recall'].append(recall(tn, fp, fn, tp))
        eval_metrics['f1_score'].append(f1_score(tn, fp, fn, tp))
        eval_metrics['iou'].append(iou(tn, fp, fn, tp))

        create_html_with_images(alpha,
                                os.path.join(gt_overlay_path, f'{img_id}_gt_overlay.png'), 
                                os.path.join(pred_overlay_path, f'{img_id}_nhfd_overlay.png'), 
                                os.path.join(comparison_path, f'{img_id}_comparison.html'))

    #cm_df = pd.DataFrame(
    #    [[total_tn, total_fp], [total_fn, total_tp]],
    #    index=['actual background', 'actual man made'],
    #    columns=['pred background', 'pred man made']
    #)

    metrics_df = pd.DataFrame(eval_metrics)

    macro_df = metrics_df.mean(numeric_only=True).to_frame(name='value')

    micro_eval_metrics = {
        'accuracy': accuracy(total_tn, total_fp, total_fn, total_tp),
        'precision': precision(total_tn, total_fp, total_fn, total_tp),
        'recall': recall(total_tn, total_fp, total_fn, total_tp),
        'f1_score': f1_score(total_tn, total_fp, total_fn, total_tp),
        'iou': iou(total_tn, total_fp, total_fn, total_tp)
    }
    micro_df = pd.DataFrame.from_dict(micro_eval_metrics, orient='index', columns=['value'])

    
    #print("\n=== Total Confusion Matrix ===")
    #print(cm_df)

    print('\n=== Macro Overall Metrics ===')
    print(macro_df)

    print("\n=== Micro Overall Metrics ===")
    print(micro_df)

    save_metrics_as_html(metrics_df, macro_df, micro_df, alpha, os.path.join(output_path, 'eval_metrics_table.html'))

    #cm_df.to_csv(os.path.join(output_path, 'total_confusion_matrix.csv'))
    metrics_df.to_csv(os.path.join(output_path, 'eval_metrics.csv'), index=False)
    macro_df.to_csv(os.path.join(output_path, 'macro_eval_metrics.csv'), header=['value'], index_label='eval metric')
    micro_df.to_csv(os.path.join(output_path, 'micro_eval_metrics.csv'), header=['value'], index_label='eval metric')


if __name__ == "__main__":
    main()
