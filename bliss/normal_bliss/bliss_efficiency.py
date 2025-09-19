from typing import List
import matplotlib.pyplot as plt
import numpy as np


def bliss_reducible(tuple_list: List[tuple], smaller_body):
    smaller_bodies = []
    larger_bodies = []
    for term in tuple_list:
        if len(term) == smaller_body * 2:
            smaller_bodies.append(term)
        elif len(term) == (smaller_body + 1) * 2:
            larger_bodies.append(term)
    bliss_reducible_larger_bodies = []

    for larger_body in larger_bodies:
        unique_tuples = []
        seen_first_elements = set()

        for t in larger_body:
            if t[0] not in seen_first_elements:
                unique_tuples.append(t)
                seen_first_elements.add(t[0])

        if not len(unique_tuples) == (smaller_body + 1) * 2:
            bliss_reducible_larger_bodies.append(larger_body)
    reducible_combs = {}

    usable_terms = []

    for smaller_tuple in smaller_bodies:
        for larger_tuple in bliss_reducible_larger_bodies:
            set_4 = set(smaller_tuple)
            set_6 = set(larger_tuple)

            # Check if all elements in set_4 exist in set_6
            is_subset = set_4.issubset(set_6)
            if is_subset:
                difference_list = list(larger_tuple)
                for item in list(smaller_tuple):
                    if item in difference_list:
                        difference_list.remove(item)
                first_elem_index = difference_list[0][0]
                second_elem_index = difference_list[1][0]
                if first_elem_index == second_elem_index:
                    usable_terms.append(larger_tuple)
                    if smaller_tuple not in reducible_combs:
                        reducible_combs[smaller_tuple] = [larger_tuple]
                    else:
                        reducible_combs[smaller_tuple].append(larger_tuple)

    return reducible_combs, set(usable_terms)


def get_total_combs(tuple_list: List[tuple], smaller_body):
    smaller_bodies = []
    larger_bodies = []
    for term in tuple_list:
        if len(term) == smaller_body * 2:
            smaller_bodies.append(term)
        elif len(term) == (smaller_body + 1) * 2:
            larger_bodies.append(term)

    smaller_len = len(smaller_bodies)
    larger_len = len(larger_bodies)

    return smaller_len * larger_len


if __name__ == '__main__':
    values = [
        372,
        372,
        244,
        322,
        322,
        352,
        322,
        322,
        272,
        372,
        304
    ]
    indices = np.arange(len(values))  # X-axis positions

    # Plot bar chart
    plt.bar(indices, values, color='green', alpha=0.7, label='[H, A] - K')

    # Add horizontal line
    threshold = 1344
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                label=f'H - K ({threshold})')

    # Labels and title
    plt.xlabel("Different Anti-Hermitian Operators")
    plt.ylabel("# of Killer Accessible Higher-body terms")
    plt.title("")
    plt.xticks(indices)  # Show all indices
    plt.legend()

    # Show plot
    plt.show()

