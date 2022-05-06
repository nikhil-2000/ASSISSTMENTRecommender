

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

"""
0.064643744	0.017975019	0.016265089
0.064047118	0.016743506	0.015683054
0.008177837	0.001825088	0.001923777
"""
def draw_graph(lines):
    data = pd.DataFrame(columns = ["Model","Metric","Value"])
    # lines = [['GCN', 'MRR', 0.069782575], ['NN', 'MRR', 0.064643744	], ['PES', 'MRR', 0.064047118], ['Random', 'MRR', 0.008177837],
    #          ['GCN', 'AP', 	0.024664706], ['NN', 'AP', 0.017975019], ['PES', 'AP', 0.016743506], ['Random', 'AP', 0.001825088],

    for i,line in enumerate(lines):
        data.loc[i] = line

    g = sns.catplot(x="Model", y="Value", col="Metric",
                    data=data,kind="bar")
    (g.set_axis_labels("", "Metric")
      # .set_xticklabels(["MRR", "AP", "RBU"])
      .set_titles("{col_name}")
      .despine(left=True))

    plt.show()


#          ['GCN', 'RBU', 0.02482034], ['NN', 'RBU', 0.016265089], ['PES', 'RBU', 0.015683054], ['Random', 'RBU', 0.001923777]]
assistment = [['GCN', 'MRR', 0.069782575], ['NN', 'MRR', 0.05981785], ['PES', 'MRR', 0.064047118], ['Random', 'MRR', 0.008177837],
         ['GCN', 'AP', 	0.024664706], ['NN', 'AP', 0.017805382], ['PES', 'AP', 0.016743506], ['Random', 'AP', 0.001825088],
         ['GCN', 'RBU', 0.02482034], ['NN', 'RBU', 0.015731646], ['PES', 'RBU', 0.015683054], ['Random', 'RBU', 0.001923777]]

movielens = [['GCN', 'MRR', 0.330649008], ['NN', 'MRR', 0.065248005], ['PES', 'MRR', 0.088544387], ['Random', 'MRR', 0.027566092],
         ['GCN', 'AP', 	0.125583858], ['NN', 'AP', 0.01655468], ['PES', 'AP', 0.026725149], ['Random', 'AP', 0.005005436],
         ['GCN', 'RBU', 0.125132556], ['NN', 'RBU', 0.016260163], ['PES', 'RBU', 0.028160716], ['Random', 'RBU', 0.004241782]]

draw_graph(assistment)
# draw_graph(movielens)
