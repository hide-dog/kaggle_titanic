import numpy
import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

def main():
    f_in = ["corr_history_test", "corr_history_train"]

    for i in range(len(f_in)):
        y = numpy.loadtxt(f_in[i], skiprows=0, delimiter = " ", unpack=True)
        
        s = len(y)
        x = numpy.zeros(s)
        for j in range(s):
            x[j] = j
        #end

        pyplot.scatter(x, y, s=20, c='b', marker='o')


        """ glid line """
        pyplot.grid(color="gray", linestyle="dotted", which="both")

        """ axis name """
        x_axis = 'iteration'
        y_axis = 'rate of correct answer'
        pyplot.xlabel(x_axis, fontsize=18)
        pyplot.ylabel(y_axis, fontsize=18)

        # display range
        pyplot.ylim(0.0, 1.0)

        """ scale name size """
        pyplot.tick_params(labelsize=14)

        """ format adjustment """
        pyplot.tight_layout()

        """ save """
        pic_name = f_in[i] + '.png'
        pyplot.savefig(pic_name)

        """ reset """
        pyplot.clf()
    #end
# end
#
#
# ------------------------------------
main()
#
# ------------------------------------




"""
# logarithmic axis
pyplot.xscale("log")
pyplot.yscale("log")

# glid line
pyplot.grid(color="gray",linestyle="dotted", which="both")

# specify the aspect ratio in inches
pyplot.figure(figsize=(4, 6))

# display range
pyplot.xlim(0.0, 5.0)
pyplot.ylim(0.0, 10.0)

# line style
pyplot.linestyle(solid)

# legend
pyplot.plot(label='test')
pyplot.legend()

# reset
pyplot.clf()
"""
