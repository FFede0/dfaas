from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
from parse import parse
import pandas as pd
import os


def load_all_results(base_folder: str) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
  ]:
  all_run_best_metrics = pd.DataFrame()
  all_best_techniques = pd.DataFrame()
  all_all_run_results = pd.DataFrame()
  all_full_dataset_metrics = pd.DataFrame()
  all_exec_time = pd.DataFrame()
  for foldername in os.listdir(base_folder):
    if foldername.startswith("OUTPUT_"):
      day, hour = foldername.split("_")[1:]
      # parse results file
      results_filename = os.path.join(base_folder, foldername, "results.txt")
      with open(results_filename, "r") as istream:
        print(f"  {results_filename}")
        lines = istream.readlines()
        (
          run_best_metrics, 
          best_techniques, 
          all_run_results, 
          full_dataset_metrics, 
          exec_time
        ) = parse_result_file(lines)
        run_best_metrics["exp_day_hour"] = [f"{day}-{hour}"] * len(
          run_best_metrics
        )
        best_techniques["exp_day_hour"] = [f"{day}-{hour}"] * len(
          best_techniques
        )
        all_run_results["exp_day_hour"] = [f"{day}-{hour}"] * len(
          all_run_results
        )
        full_dataset_metrics["exp_day_hour"] = [f"{day}-{hour}"] * len(
          full_dataset_metrics
        )
        exec_time = pd.DataFrame({
          "exec_time": [exec_time],
          "exp_day_hour": [f"{day}-{hour}"]
        })
        # concat
        all_run_best_metrics = pd.concat(
          [all_run_best_metrics, run_best_metrics], ignore_index = True
        )
        all_best_techniques = pd.concat(
          [all_best_techniques, best_techniques], ignore_index = True
        )
        all_all_run_results = pd.concat(
          [all_all_run_results, all_run_results], ignore_index = True
        )
        all_full_dataset_metrics = pd.concat(
          [all_full_dataset_metrics, full_dataset_metrics], ignore_index = True
        )
        all_exec_time = pd.concat(
          [all_exec_time, exec_time], ignore_index = True
        )
      # load predictions
      predictions = pd.read_csv(
        os.path.join(base_folder, f"PREDICT_{day}_{hour}", "prediction.csv")
      )
      # plot predictions and MAPE
      plot_predictions_and_mape(
        predictions, os.path.join(base_folder, f"PREDICT_{day}_{hour}")
      )
  # save
  all_run_best_metrics.to_csv(
    os.path.join(base_folder, "run_best_metrics.csv"), index = False
  )
  all_best_techniques.to_csv(
    os.path.join(base_folder, "best_techniques.csv"), index = False
  )
  all_all_run_results.to_csv(
    os.path.join(base_folder, "all_run_results.csv"), index = False
  )
  all_full_dataset_metrics.to_csv(
    os.path.join(base_folder, "full_dataset_metrics.csv"), index = False
  )
  all_exec_time.to_csv(
    os.path.join(base_folder, "exec_time.csv"), index = False
  )
  # return
  return (
    all_run_best_metrics,
    all_best_techniques,
    all_all_run_results,
    all_full_dataset_metrics,
    all_exec_time
  )


def parse_result_file(lines: list):
  run_best_metrics = pd.DataFrame()
  best_techniques = {
    "run": [], 
    "metric": [], 
    "technique": [],
    "configuration": []
  }
  all_run_results = {
    "run": [], 
    "technique": [],
    "configuration": []
  }
  full_dataset_metrics = pd.DataFrame()
  exec_time = None
  done = False
  line_idx = 0
  while not done:
    line = lines[line_idx]
    # parse run best results
    if line.startswith("Printing results for run"):
      run_idx = int(parse("Printing results for run {}\n", line)[0])
      # load all-techniques metrics
      line_idx += 2
      metrics = {
        "technique": [],
        "metric": [],
        "train": [],
        "hp_selection": [],
        "val": []
      }
      while not lines[line_idx].startswith("Overall best result"):
        technique, metric, train, hp_selection, val = parse(
          "Technique.{} [{}]: (Training {} - HP Selection {}) - Validation {}",
          lines[line_idx].strip()
        )
        metrics["technique"].append(technique.strip())
        metrics["metric"].append(metric)
        metrics["train"].append(float(train))
        metrics["hp_selection"].append(float(hp_selection))
        metrics["val"].append(float(val))
        # move to next line
        line_idx += 1
      metrics = pd.DataFrame(metrics)
      metrics["run"] = [run_idx] * len(metrics)
      run_best_metrics = pd.concat(
        [run_best_metrics, metrics], ignore_index = True
      )
      # load best result info
      metric, technique, configuration = parse(
        "Overall best result (according to {}) is Technique.{}, with configuration [{}]\n",
        lines[line_idx]
      )
      best_techniques["run"].append(run_idx)
      best_techniques["metric"].append(metric)
      best_techniques["technique"].append(technique)
      best_techniques["configuration"].append(f"[{configuration}]")
      # load best result metrics
      line_idx += 3
      while not lines[line_idx].startswith("<--"):
        metric, train, hp_selection, val = parse(
          "{}: (Training {} - HP Selection {}) - Validation {}",
          lines[line_idx].strip()
        )
        if f"{metric}_train" not in best_techniques:
          best_techniques[f"{metric}_train"] = [float(train)]
        else:
          best_techniques[f"{metric}_train"].append(float(train))
        if f"{metric}_hp_selection" not in best_techniques:
          best_techniques[f"{metric}_hp_selection"] = [float(hp_selection)]
        else:
          best_techniques[f"{metric}_hp_selection"].append(float(hp_selection))
        if f"{metric}_val" not in best_techniques:
          best_techniques[f"{metric}_val"] = [float(val)]
        else:
          best_techniques[f"{metric}_val"].append(float(val))
        # move to next line
        line_idx += 1
    # parse run detailed results
    elif line.startswith("Run "):
      run_idx, technique, configuration, metric, hp_selection, _, val = parse(
        "Run {} - Technique {} - Conf [{}] - Training {} {} - Test {} {}\n",
        line
      )
      all_run_results["run"].append(int(run_idx))
      all_run_results["technique"].append(technique)
      all_run_results["configuration"].append(f"[{configuration}]")
      if f"{metric}_hp_selection" not in all_run_results:
        all_run_results[f"{metric}_hp_selection"] = [float(hp_selection)]
      else:
        all_run_results[f"{metric}_hp_selection"].append(float(hp_selection))
      if f"{metric}_val" not in all_run_results:
        all_run_results[f"{metric}_val"] = [float(val)]
      else:
        all_run_results[f"{metric}_val"].append(float(val))
    # parse metrics on full dataset
    elif line.startswith("Validation metrics on full dataset"):
      line_idx += 1
      while not lines[line_idx].startswith("Built the final regressors"):
        technique, temp = lines[line_idx].split(":")
        fdm = {"technique": [parse("Technique.{}", technique.strip())[0]]}
        tokens = temp.split(" -")
        for token_idx, token in enumerate(tokens):
          if token_idx < len(tokens) - 1:
            metric, val = parse(" {} {}", token)
            fdm[metric] = [val]
          else:
            metric, val = parse(" {} {}\n", token)
            fdm[metric] = [val]
        full_dataset_metrics = pd.concat(
          [full_dataset_metrics, pd.DataFrame(fdm)], ignore_index = True
        )
        # move to the next line
        line_idx += 1
    # parse execution time
    elif line.startswith("Execution Time :"):
      exec_time = float(parse("Execution Time : {}\n", line)[0])
      done = True
    # move to next line
    line_idx += 1
  best_techniques = pd.DataFrame(best_techniques)
  all_run_results = pd.DataFrame(all_run_results)
  return (
    run_best_metrics, 
    best_techniques, 
    all_run_results, 
    full_dataset_metrics, 
    exec_time
  )


def plot_predictions_and_mape(
    predictions: pd.DataFrame, foldername: str = None
  ):
  # compute MAPE
  mape = (
    abs(predictions["real"] - predictions["pred"]) / predictions["real"]
  ) * 100
  # -- predictions
  _, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 5))
  predictions.plot.scatter(
    x = "real",
    y = "pred",
    ax = axs[0],
    grid = True
  )
  p1 = max(predictions["pred"].max(), predictions["real"].max())
  p2 = min(predictions["pred"].min(), predictions["real"].min())
  axs[0].plot([p1, p2], [p1, p2], 'k--')
  # -- MAPE
  mape.plot(
    grid = True, 
    marker = ".", 
    linewidth = 0.01, 
    ax = axs[1]
  )
  axs[1].axhline(
    y = mape.mean(), 
    color = mcolors.TABLEAU_COLORS["tab:red"],
    linestyle = "dashed",
    linewidth = 2
  )
  axs[1].set_ylabel("MAPE")
  if foldername is not None:
    plt.savefig(
      os.path.join(foldername, "predictions_and_mape.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.show()


def train_test_split():
  data = pd.read_csv("output/output-energy/all_results_no_outliers.csv")
  # train/test split
  data_test = pd.DataFrame()
  data_train = pd.DataFrame()
  for key, group in data.groupby("node_type"):
    test = group.sample(frac = 0.2)
    train = group.drop(test.index)
    data_test = pd.concat([data_test, test], ignore_index = True)
    data_train = pd.concat([data_train, train], ignore_index = True)
  # save
  data_test.to_csv("output/output-energy/all_results_no_outliers_TEST.csv")
  data_train.to_csv("output/output-energy/all_results_no_outliers_TRAIN.csv")


if __name__ == "__main__":
  base_folder = "/Users/federicafilippini/Documents/GitHub/FORKs/aMLLibrary/DFAAS"
  targets = ["cpu", "ram", "power"]
  exp_types = ["NoAugmentation", "AugmentationAndSelection"]
  techniques = {
    "STEPWISE": "Stepwise",
    "XGBOOST": "XGBoost",
    "DT": "DecisionTree",
    "RF": "RandomForest",
    "SVR": "SVR",
    "LR_RIDGE": "LRRidge",
    "NEURAL_NETWORK": "NeuralNetwork"
  }
  train_test_metrics = pd.DataFrame()
  for target in targets:
    print(80*"#")
    print(target)
    for exp_type in exp_types:
      print(80*"-")
      print(exp_type)
      exp_folder = os.path.join(base_folder, f"OUTPUT_{target}Node", exp_type)
      (
        all_run_best_metrics,
        all_best_techniques,
        all_all_run_results,
        all_full_dataset_metrics,
        all_exec_time
      ) = load_all_results(exp_folder)
      # add test metrics
      test_mape = []
      for technique, exp_day_hour in zip(
          all_full_dataset_metrics["technique"], 
          all_full_dataset_metrics["exp_day_hour"]
        ):
        day, hour = exp_day_hour.split("-")
        technique_name = techniques[technique]
        with open(
            os.path.join(
              exp_folder, f"PREDICT_{day}_{hour}", technique_name, "mape.txt"
            ), 
            "r"
          ) as istream:
          test_mape.append(float(istream.readlines()[0]))
      all_full_dataset_metrics["MAPE_test"] = test_mape
      all_full_dataset_metrics["target"] = [target] * len(
        all_full_dataset_metrics
      )
      all_full_dataset_metrics["exp_type"] = [exp_type] * len(
        all_full_dataset_metrics
      )
      # concat
      train_test_metrics = pd.concat(
        [train_test_metrics, all_full_dataset_metrics], ignore_index = True
      )
  # save
  train_test_metrics.to_csv(
    os.path.join(base_folder, "train_test_metrics.csv"), index = False
  )
  # all_best_techniques.sort_values(by = "MAPE_hp_selection")
