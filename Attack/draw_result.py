import matplotlib.pyplot as plt
import numpy as np

MI_1=[0.94,0.92,0.927,0.824,0.788,0.826,0.898,0.823]
O_1 = [0.988, 0.991, 0.981, 0.973, 0.928, 0.984, 0.741, 0.977]
MI_2 = [0.912, 0.899, 0.902, 0.793, 0.691, 0.821, 0.613, 0.823]
O_2 = [0.923, 0.957, 0.968, 0.860, 0.944, 0.975, 0.640, 0.941]
MI_3 = [0.887, 0.867, 0.975, 0.821, 0.832, 0.792, 0.821, 0.844]
O_3 = [0.991, 0.975, 0.988, 0.865, 0.989, 0.978, 0.841, 0.989]
data1 = [MI_1,O_1]
data2 = [MI_2,O_2]
data3 = [MI_3,O_3]

err_MI_1 = [0.024,0.003,0.015,0.011,0.006,0.007,0.023,0.025]
err_O_1 = [0.003,0.013,0.007,0.005,0.003,0.006,0.021,0.004]
err_MI_2 = [0.021,0.015,0.008,0.011,0.009,0.013,0.023,0.036]
err_O_2 = [0.009,0.012,0.003,0.010,0.008,0.002,0.009,0.008]
err_MI_3 = [0.019,0.011,0.009,0.013,0.017,0.021,0.017,0.028]
err_O_3 = [0.013,0.007,0.003,0.003,0.005,0.009,0.012,0.003]
barWidth = 0.25
fig, axs = plt.subplots(1, 3)

br1 = np.arange(8)
br2 = [x + barWidth for x in br1]

 
# Make the plot

axs[0].bar(br1, data1[0], color ='b', width = barWidth,
        edgecolor ='grey', label ='MDI(With WGAN effects)')

axs[0].bar(br2, data1[1], color ='g', width = barWidth,
        edgecolor ='grey', label ='MDI(WGAN effects eliminated)')

axs[0].errorbar(br1, data1[0], yerr=err_MI_1, fmt="o", color="r")
axs[0].errorbar(br2, data1[1], yerr=err_O_1, fmt="o", color="r")
axs[0].set_title("NDCG@1")
axs[1].bar(br1, data2[0], color ='b', width = barWidth,
        edgecolor ='grey', label ='MI')

axs[1].bar(br2, data2[1], color ='g', width = barWidth,
        edgecolor ='grey', label ='Our MDI')
axs[1].errorbar(br1, data2[0], yerr=err_MI_2, fmt="o", color="r")
axs[1].errorbar(br2, data2[1], yerr=err_O_2, fmt="o", color="r")
axs[1].set_title("NDCG@2")

axs[2].bar(br1, data3[0], color ='b', width = barWidth,
        edgecolor ='grey', label ='MI')
axs[2].bar(br2, data3[1], color ='g', width = barWidth,
        edgecolor ='grey', label ='Our MDI')
axs[2].errorbar(br1, data3[0], yerr=err_MI_3, fmt="o", color="r")
axs[2].errorbar(br2, data3[1], yerr=err_O_3, fmt="o", color="r")
axs[2].set_title("NDCG@3")

plt.setp(axs, xticks=range(8), xticklabels=['M', 'E', 'FM', 'C', 'L','Em','C10','C100'],
        yticks=[0.2,0.4,0.6,0.8,1])

 
# Adding Xticks
fig.text(0.5, 0.04, 'Target Dataset', ha='center')
fig.text(0.04, 0.5, 'NDCG Score', va='center', rotation='vertical')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
'''
data = [347.6,382.9,357.3,322.8,376.5,377.1,365.9,352.6]
label=['M', 'E', 'FM', 'C', 'L','Em','C10','C100']

c = [28.7,26.3,23.5,27.9,27.2,25.9,24.3,22.2]

plt.bar(label, data)



plt.errorbar(label, data, yerr=c, fmt="o", color="r")

plt.xlabel("Target Dataset")
plt.ylabel("Distance")
#plt.xticks(range(8),label=['M', 'E', 'FM', 'C', 'L','Em','C10','C100'])
plt.show()
'''
