import numpy as np
import matplotlib.pyplot as plt
from neuron import r_m
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import ConnectionPatch

# Matplotlib visualization of the HD calibration circuit components

class PolarPlotter:
    def __init__(self, n, bias, invert,PosCueDirEncoder):
        self.showStims = False
        self.bias = bias
        self.invert = invert
        self.n = n
        plt.ion()

        # Add figure with subplots,spacing, axe labels & titles
        self.fig = plt.figure(figsize=(7,9))

        ax1 = plt.subplot(331, projection="polar")
        ax2 = plt.subplot(334, projection="polar")
        ax3 = plt.subplot(332)
        ax4 = plt.subplot(335, projection="polar")
        ax5 = plt.subplot(337)
        #ax6 = plt.subplot(339, projection="polar")
        ax7 = plt.subplot(338)

        ax7_2 = ax7.twinx()
        ax7_3 = ax7.twiny()

        self.fig.tight_layout(pad=4.5)

        ax1.set_title('Head Direction')
        ax2.set_title('Egocentric Cue Direction')
        ax2.xaxis.set_major_locator(mticker.FixedLocator(ax2.get_xticks().tolist()))
        ax2.set_xticklabels(['     Ahead', '', 'Left', '', 'Behind     ', '', 'Right', ''])
        ax3.set_title('Conjunctive Cell Field 1')
        ax3.set_xlabel("Egocentric Cue Direction (deg)")
        ax3.set_ylabel("Head Direction (deg)")
        ax4.set_title("Allocentric Cue Direction")
        ax5.set_title('Conjunctive Cell Field 2')
        ax5.set_xlabel("Egocentric Cue Direction (deg)")
        ax5.set_ylabel("Allocentric Cue Direction (deg)")
        #ax6.set_title("CONJ -> HDC Feedback Signal Visualization")
        ax7.set_title("Allocentric Cue Directions at Positions")
        ax7.set_xlabel("Horizontal Matrix Dimension")
        ax7.set_ylabel("Vertical Matrix Dimension")
        ax7_2.set_ylabel('Real Vertical Cue Distance in Units')
        ax7_3.set_xlabel('Real Horizontal Distance in Units')

        self.matrix_conj_rates = np.zeros((n,n))
        self.matrix_conj_rates_2 = np.zeros((n,n))

        self.img = ax3.imshow(self.matrix_conj_rates, cmap='hot', interpolation='nearest',  vmin=0,
                         vmax=10,extent=[0, 360, 0, 360])
        self.img_2 = ax5.imshow(self.matrix_conj_rates_2, cmap='hot', interpolation='nearest',  vmin=0,
                         vmax=10,extent=[360, 0, 0, 360])

        divider3 = make_axes_locatable(ax3)
        divider5 = make_axes_locatable(ax5)

        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cax5 = divider5.append_axes("right", size="5%", pad=0.05)

        cbar3 = self.fig.colorbar(self.img, cax=cax3)
        cbar5 = self.fig.colorbar(self.img_2, cax=cax5)

        cbar3.set_label('Firing Rate (Hz)', rotation=90)
        cbar5.set_label('Firing Rate (Hz)', rotation=90)

        matrixLength = (len(PosCueDirEncoder.posACDMatrix))
        matDimReal = PosCueDirEncoder.matDimReal/2

        vecFieldSpace = np.linspace(-(matrixLength/2), (matrixLength/2), matrixLength)
        vecFieldSpace2 = np.linspace((matrixLength/2), -(matrixLength/2), matrixLength)

        self.x, self.y = np.meshgrid(vecFieldSpace, vecFieldSpace2)

        self.u = PosCueDirEncoder.posACDMatrix[:,:,0]
        self.v = PosCueDirEncoder.posACDMatrix[:,:,1]
        self.vecField = ax7.quiver(self.x, self.y, self.u, self.v,scale=20,color='black',width=0.0075)
        ax7_2.set(ylim=(-matDimReal, matDimReal))
        ax7_3.set(xlim=(-matDimReal, matDimReal))

        if self.showStims:
            plt.title("Shift right stimulus: {:3.2f}\nShift left stimulus: {:.2f}".format(0.0, 0.0))

        x = self.makeXSpace(n, bias, invert)

        # plot returns a list of Line2D objects.
        # To access the first object of the list, the comma operator is used: e.gg. line1,
        self.line1, = ax1.plot(x, [r_m] * (n + 1), 'r-', label="HDC layer")
        #self.line2, = ax1.plot(x, [r_m] * (n + 1), 'b-', label="shift-left layer")
        #self.line3, = ax1.plot(x, [r_m] * (n + 1), 'g-', label="shift-right layer")
        self.line4, = ax2.plot(x, [r_m] * (n + 1), 'c-', label="ECD layer")
        self.line7, = ax4.plot(x, [r_m] * (n + 1), 'c-', label="ACD layer")
        #self.line8, = ax6.plot(x, [20] * (n + 1), 'r-', label="HDC feedback signal")

        # compass True orientation
        self.line5, = ax1.plot([np.pi, np.pi], [0.0, r_m], 'k-', label="True orientation")
        # compass Decoded orientation
        self.line6, = ax1.plot([np.pi, np.pi], [0.0, r_m], 'm-', label="Decoded orientation")

        ax1.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
        ax2.legend()
        ax4.legend()
        #ax6.legend()

        plt.show()

    def makeXSpace(self, n, bias, invert):
        return [(x + bias) * (-1 if invert else 1) % (np.pi * 2) for x in np.linspace(0.0, np.pi * 2, (n+1))]
    def plot(self, rates_hdc, rates_sl, rates_sr, rates_ecd, rates_conj, rates_acd, rates_conj_2, rates_hdc2,
             PosCueDirEncoder,stimL, stimR, trueOrientation, decOrientation): #rates_hdc2,
        n = self.n
        if self.showStims:
            plt.title("Shift right stimulus: {:3.2f}\nShift left stimulus: {:.2f}".format(stimR, stimL))
        l1 = rates_hdc
        l1.append(rates_hdc[0])
        l2 = rates_sl
        l2.append(rates_sl[0])
        l3 = rates_sr
        l3.append(rates_sr[0])
        l4 = rates_ecd
        l4.append(rates_ecd[0])
        l7 = rates_acd
        l7.append(rates_acd[0])
        l8 = rates_hdc2
        l8.append(rates_hdc2[0])
        self.line1.set_ydata(l1)
        #self.line2.set_ydata(l2)
        #self.line3.set_ydata(l3)
        self.line4.set_ydata(l4)
        self.line7.set_ydata(l7)
        #self.line8.set_ydata(l8)

        tOr = (trueOrientation + self.bias) * (-1 if self.invert else 1)
        dOr = (decOrientation + self.bias) * (-1 if self.invert else 1)
        self.line5.set_xdata([tOr, tOr])
        self.line6.set_xdata([dOr, dOr])

        self.img.set_data(np.array(rates_conj).reshape(n, n))
        #self.img.autoscale()

        self.img_2.set_data(np.array(rates_conj_2).reshape(n, n))
        #self.img_2.autoscale()

        self.u = np.multiply(PosCueDirEncoder.posACDMatrix[:, :, 0],1)
        self.v = np.multiply(PosCueDirEncoder.posACDMatrix[:, :, 1],1)
        self.vecField.set_UVC(self.u,self.v)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()