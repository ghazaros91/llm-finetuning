from datasets import load_metric

def evaluate_model(model, eval_dataset, metric_name="accuracy"):
    """
    Evaluate the model on the evaluation dataset.

    Args:
        model: The model to be evaluated.
        eval_dataset: The evaluation dataset.
        metric_name (str): The metric to be used for evaluation.

    Returns:
        float: The evaluation score.
    """
    metric = load_metric(metric_name)
    predictions = model.predict(eval_dataset)
    metric.add_batch(predictions=predictions, references=eval_dataset["labels"])
    return metric.compute()
