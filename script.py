import pandas as pd
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tqdm.auto import tqdm
from fsrs_optimizer import (
    lineToTensor,
    next_interval,
    power_forgetting_curve,
    FSRS,
    DEFAULT_PARAMETER,
)
import multiprocessing as mp
from functools import partial
import matplotlib

matplotlib.use("Agg")

plt.style.use("ggplot")
cmap = colormaps.get_cmap("tab20")

desired_retention = 0.8
deck_size = 20000

learn_limit_perday = 20
review_limit_perday = 80

learn_span = int(deck_size / learn_limit_perday)

first_rating_prob = np.array([0.24, 0.094, 0.495, 0.171])
review_rating_prob = np.array([0.224, 0.631, 0.145])
review_costs = np.array([23.0, 11.68, 7.33, 5.6])
learn_costs = np.array([33.79, 24.3, 13.68, 6.5])

seed = 42
moving_average_win_size = 30


feature_list = [
    "difficulty",
    "stability",
    "retrievability",
    "interval",
    "delta_t",
    "reps",
    "lapses",
    "last_date",
    "due",
    "r_history",
    "t_history",
    "p_history",
    "states",
    "time",
    "R(tomorrow)",
    "PRL",  # Potential Retrievability Loss (PRL) = R(Today) - R(Tomorrow)
    "PSG",  # Potential Stability Gain (PSG) = S(recall)/S(today) * R(Today)
]
field_map = {key: i for i, key in enumerate(feature_list)}


def generate_rating(review_type):
    if review_type == "new":
        return np.random.choice([1, 2, 3, 4], p=first_rating_prob)
    elif review_type == "recall":
        return np.random.choice([2, 3, 4], p=review_rating_prob)


class Collection:
    def __init__(self):
        self.model = FSRS(DEFAULT_PARAMETER)
        self.model.eval()

    def states(self, t_history, r_history):
        with torch.no_grad():
            line_tensor = lineToTensor(
                list(zip([str(t_history)], [str(r_history)]))[0]
            ).unsqueeze(1)
            output_t = self.model(line_tensor)
            return output_t[-1][0]

    def next_states(self, states, t, r):
        with torch.no_grad():
            return self.model.step(torch.FloatTensor([[t, r]]), states.unsqueeze(0))[0]

    def init(self):
        t = 0
        r = generate_rating("new")
        p = round(first_rating_prob[r - 1], 2)
        new_states = self.states(t, r)
        return r, t, p, new_states


def moving_average(data, window_size=moving_average_win_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode="valid")


review_sorting_orders = [
    "due_date_asc",
    "interval_asc",
    "interval_desc",
    "difficulty_asc",
    "difficulty_desc",
    "retrievability_asc",
    "retrievability_desc",
    "stability_asc",
    "stability_desc",
    "random",
    "add_order_asc",
    "add_order_desc",
    "PRL_desc",
    "PSG_desc",
]


def run_simulation(seed, review_sorting_order):
    np.random.seed(seed)
    random.seed(seed)

    new_card_per_day = np.array([0] * learn_span)
    review_card_per_day = np.array([0.0] * learn_span)
    time_per_day = np.array([0.0] * learn_span)
    learned_per_day = np.array([0.0] * learn_span)
    retention_per_day = np.array([0.0] * learn_span)
    expected_memorization_per_day = np.array([0.0] * learn_span)

    card = pd.DataFrame(
        np.zeros((deck_size, len(feature_list))),
        index=range(deck_size),
        columns=feature_list,
    )
    card["states"] = card["states"].astype(object)
    card["r_history"] = card["r_history"].astype(str)
    card["t_history"] = card["t_history"].astype(str)
    card["p_history"] = card["p_history"].astype(str)
    card["reps"] = 0
    card["lapses"] = 0
    card["due"] = learn_span

    student = Collection()

    for day in tqdm(range(learn_span)):
        reviewed = 0
        learned = 0
        review_time_today = 0
        learn_time_today = 0

        card["delta_t"] = day - card["last_date"]
        card["retrievability"] = power_forgetting_curve(
            card["delta_t"], card["stability"]
        )
        if review_sorting_order.startswith("PRL"):
            card["R(tomorrow)"] = power_forgetting_curve(
                card["delta_t"] + 1, card["stability"]
            )
            card["PRL"] = card["retrievability"] - card["R(tomorrow)"]
        if review_sorting_order.startswith("PSG"):
            card["S(recall)"] = card.apply(
                lambda row: (
                    float(student.next_states(row["states"], row["delta_t"], 3)[0])
                    if row["stability"] != 0
                    else 0
                ),
                axis=1,
            )
            # card["S(forget)"] = card.apply(
            #     lambda row: (
            #         float(student.next_states(row["states"], row["delta_t"], 1)[0])
            #         if row["stability"] != 0
            #         else 0
            #     ),
            #     axis=1,
            # )
            card["PSG"] = (
                card["S(recall)"]
                * card["retrievability"]
                # + card["S(forget)"] * (1 - card["retrievability"])
            ) / card["stability"].map(lambda x: x if x != 0 else 1)
            # card["S(recall_tomorrow)"] = card.apply(
            #     lambda row: (
            #         float(student.next_states(row["states"], row["delta_t"] + 1, 3)[0])
            #         if row["stability"] != 0
            #         else 0
            #     ),
            #     axis=1,
            # )
            # card["S(forget_tomorrow)"] = card.apply(
            #     lambda row: (
            #         float(student.next_states(row["states"], row["delta_t"] + 1, 1)[0])
            #         if row["stability"] != 0
            #         else 0
            #     ),
            #     axis=1,
            # )
            # card["PSG_tomorrow"] = (
            #     card["S(recall_tomorrow)"] * card["retrievability"]
            #     # + card["S(forget_tomorrow)"] * (1 - card["retrievability"])
            # ) / card["stability"].map(lambda x: x if x != 0 else 1)
            # card["PLSG"] = card["PSG"] - card["PSG_tomorrow"]
        need_review = card[card["due"] <= day]

        sum_true_retrievability_today = 0

        if review_sorting_order != "random":
            column, order = review_sorting_order.rsplit("_", 1)
            ascending = order == "asc"
            if column == "add_order":
                need_review = need_review.sort_index(ascending=ascending)
            elif column == "due_date":
                need_review = need_review.sort_values("due", ascending=ascending)
            else:
                need_review = need_review.sort_values(column, ascending=ascending)
        else:
            need_review = need_review.sample(frac=1)

        for idx in need_review.index:
            if reviewed >= review_limit_perday:
                break

            reviewed += 1
            last_date = card.iat[idx, field_map["last_date"]]
            due = card.iat[idx, field_map["due"]]
            card.iat[idx, field_map["last_date"]] = day
            ivl = card.iat[idx, field_map["delta_t"]]
            card.iat[idx, field_map["t_history"]] += f",{ivl}"
            stability = card.iat[idx, field_map["stability"]]
            retrievability = card.iat[idx, field_map["retrievability"]]
            sum_true_retrievability_today += retrievability
            card.iat[idx, field_map["p_history"]] += f",{retrievability:.2f}"
            reps = card.iat[idx, field_map["reps"]]
            lapses = card.iat[idx, field_map["lapses"]]
            states = card.iat[idx, field_map["states"]]

            if random.random() < retrievability:
                rating = generate_rating("recall")
                recall_time = review_costs[rating - 1]
                review_time_today += recall_time
                card.iat[idx, field_map["r_history"]] += f",{rating}"
                new_states = student.next_states(states, ivl, rating)
                new_stability = float(new_states[0])
                new_difficulty = float(new_states[1])
                card.iat[idx, field_map["stability"]] = new_stability
                card.iat[idx, field_map["difficulty"]] = new_difficulty
                card.iat[idx, field_map["states"]] = new_states
                card.iat[idx, field_map["reps"]] = reps + 1
                card.iat[idx, field_map["time"]] += recall_time
                interval = next_interval(new_stability, desired_retention)
                card.iat[idx, field_map["interval"]] = interval
                card.iat[idx, field_map["due"]] = day + interval
            else:
                review_time_today += review_costs[0]

                rating = 1
                card.iat[idx, field_map["r_history"]] += f",{rating}"

                new_states = student.next_states(states, ivl, 1)
                new_stability = float(new_states[0])
                new_difficulty = float(new_states[1])

                card.iat[idx, field_map["stability"]] = new_stability
                card.iat[idx, field_map["difficulty"]] = new_difficulty
                card.iat[idx, field_map["states"]] = new_states

                reps = 0
                lapses = lapses + 1

                card.iat[idx, field_map["reps"]] = reps
                card.iat[idx, field_map["lapses"]] = lapses

                interval = next_interval(new_stability, desired_retention)
                card.iat[idx, field_map["interval"]] = interval
                card.iat[idx, field_map["due"]] = day + interval
                card.iat[idx, field_map["time"]] += review_costs[0]

        retention_per_day[day] = (
            sum_true_retrievability_today / reviewed if reviewed > 0 else np.nan
        )

        need_learn = card[card["stability"] == 0]

        for idx in need_learn.index:
            if learned >= learn_limit_perday:
                break
            learned += 1
            r, t, p, new_states = student.init()
            learn_time_today += learn_costs[r - 1]
            card.iat[idx, field_map["last_date"]] = day

            card.iat[idx, field_map["reps"]] = 1
            card.iat[idx, field_map["lapses"]] = 0

            new_stability = float(new_states[0])
            new_difficulty = float(new_states[1])

            card.iat[idx, field_map["r_history"]] = str(r)
            card.iat[idx, field_map["t_history"]] = str(t)
            card.iat[idx, field_map["p_history"]] = str(p)
            card.iat[idx, field_map["stability"]] = new_stability
            card.iat[idx, field_map["difficulty"]] = new_difficulty
            card.iat[idx, field_map["states"]] = new_states

            delta_t = next_interval(new_stability, desired_retention)
            card.iat[idx, field_map["due"]] = day + delta_t
            card.iat[idx, field_map["time"]] = learn_costs[r - 1]

        new_card_per_day[day] = learned
        review_card_per_day[day] = reviewed
        learned_per_day[day] = learned_per_day[day - 1] + learned
        time_per_day[day] = review_time_today + learn_time_today
        expected_memorization_per_day[day] = sum(
            card[card["retrievability"] > 0]["retrievability"]
        )

    total_learned = sum(new_card_per_day)
    total_time = sum(time_per_day)
    total_remembered = int(card["retrievability"].sum())
    average_true_retention = np.nanmean(retention_per_day)

    return {
        "new_card_per_day": new_card_per_day,
        "review_card_per_day": review_card_per_day,
        "time_per_day": time_per_day,
        "learned_per_day": learned_per_day,
        "retention_per_day": retention_per_day,
        "expected_memorization_per_day": expected_memorization_per_day,
        "stats": {
            "review_sorting_order": review_sorting_order,
            "total_learned": total_learned,
            "total_time": total_time,
            "total_remembered": total_remembered,
            "average_true_retention": average_true_retention,
        },
        "retrievability": card[card["retrievability"] > 0]["retrievability"].values,
    }


sample_size = 5


def run_simulation_wrapper(review_sorting_order, seed):
    return run_simulation(seed, review_sorting_order)


def run_multi_process_simulation():
    table_data = []
    all_results = {}

    pool = mp.Pool(processes=mp.cpu_count())

    try:
        for review_sorting_order in review_sorting_orders:
            seeds = range(seed, seed + sample_size)
            partial_run_simulation = partial(
                run_simulation_wrapper, review_sorting_order
            )
            results_list = pool.map(partial_run_simulation, seeds)

            combined_results = {
                "new_card_per_day": np.mean(
                    [r["new_card_per_day"] for r in results_list], axis=0
                ),
                "review_card_per_day": np.mean(
                    [r["review_card_per_day"] for r in results_list], axis=0
                ),
                "time_per_day": np.mean(
                    [r["time_per_day"] for r in results_list], axis=0
                ),
                "learned_per_day": np.mean(
                    [r["learned_per_day"] for r in results_list], axis=0
                ),
                "retention_per_day": np.mean(
                    [r["retention_per_day"] for r in results_list], axis=0
                ),
                "expected_memorization_per_day": np.mean(
                    [r["expected_memorization_per_day"] for r in results_list], axis=0
                ),
                "stats": {
                    "review_sorting_order": review_sorting_order,
                    "total_learned": np.mean(
                        [r["stats"]["total_learned"] for r in results_list]
                    ),
                    "total_time": np.mean(
                        [r["stats"]["total_time"] for r in results_list]
                    ),
                    "total_remembered": np.mean(
                        [r["stats"]["total_remembered"] for r in results_list]
                    ),
                    "average_true_retention": np.mean(
                        [r["stats"]["average_true_retention"] for r in results_list]
                    ),
                },
                "retrievability": np.mean(
                    [r["retrievability"] for r in results_list], axis=0
                ),
            }

            all_results[review_sorting_order] = combined_results

            table_data.append(
                [
                    review_sorting_order,
                    combined_results["stats"]["total_learned"],
                    combined_results["stats"]["total_time"],
                    combined_results["stats"]["total_remembered"],
                    combined_results["stats"]["average_true_retention"],
                ]
            )

    except Exception as e:
        print(f"error: {e}")
    finally:
        pool.close()
        pool.join()

    for review_sorting_order in review_sorting_orders:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(
            all_results[review_sorting_order]["retrievability"],
            bins=20,
            alpha=0.5,
            color=cmap(review_sorting_orders.index(review_sorting_order)),
        )
        ax.set_xlim(0, 1)
        ax.set_title(f"Average Retrievability Distribution - {review_sorting_order}")
        fig.savefig(f"retrievability_distribution_{review_sorting_order}.png")
        plt.close(fig)

    plot_titles = [
        "Review Count per Day",
        "Time Cost in minutes per Day",
        "Cumulative Learn Count per Day",
        "True Retention per Day",
        "Memorized Count per Day",
    ]

    for i, data_key in enumerate(
        [
            "review_card_per_day",
            "time_per_day",
            "learned_per_day",
            "retention_per_day",
            "expected_memorization_per_day",
        ]
    ):
        plt.figure(figsize=(10, 6))
        for j, review_sorting_order in enumerate(review_sorting_orders):
            data = all_results[review_sorting_order][data_key]
            if data_key in ["review_card_per_day", "time_per_day", "retention_per_day"]:
                data = moving_average(data)
            if data_key == "time_per_day":
                data = data / 60
            plt.plot(data, label=review_sorting_order, color=cmap(j))

        plt.title(f"{plot_titles[i]} ({moving_average_win_size} days average)")
        plt.legend()
        plt.savefig(f"{data_key}_comparison_{review_sorting_order}.png")
        plt.close()

    df = pd.DataFrame(
        table_data,
        columns=[
            "order",
            "total_learned",
            "total_time",
            "total_remembered",
            "average_true_retention",
        ],
    )
    df["seconds_per_remembered_card"] = (
        df["total_time"] / df["total_remembered"]
    ).round(1)
    df["average_true_retention"] = df["average_true_retention"].round(3)
    df.sort_values(by=["seconds_per_remembered_card"], inplace=True)
    df.to_csv("simulator-result.tsv", index=False, sep="\t")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    run_multi_process_simulation()
