import matplotlib.pyplot as plt
from matplotlib.table import table
import numpy as np
import random
import time
from scipy import stats
import copy
from sortedcontainers import SortedList
import math

class Rectangle:
    
    def __init__(self, num, x1, y1, x2, y2):
        self.rectNum = num
        self.bottom_left = (x1, y1)
        self.top_right = (x2, y2)
        self.adjacencies = []

    def equals(self, other):
        if (self.bottom_left == other.bottom_left) and (self.top_right == other.top_right):
            return True
        else:
            return False

    def toString(self):
        out = str(self.rectNum) + ', ' + str(len(self.adjacencies)) + ', '
        for r in self.adjacencies:
            out += str(r.rectNum) + ', ' + str(r.bottom_left[0]) + ', ' + str(
                r.bottom_left[1]) + ', ' + str(r.top_right[1]) + ', '
        print(out)


def checkOverlap(bottom_left, top_right, Rectangles):
    for r in Rectangles: 
        if (bottom_left[0] < r.top_right[0] and top_right[0] > r.bottom_left[0] 
        and bottom_left[1] < r.top_right[1] and top_right[1] > r.bottom_left[1]):
            return True
    return False

    
def genRects(numrect, gridSize, maxArea):
    Rectangles = []
    for i in range(numrect):
        Area = maxArea + 1
        overlap = True

        while ((Area > maxArea) or (overlap == True)):
            bottom_left = (random.randint(0, gridSize), random.randint(0, gridSize))
            top_right = (random.randint(0, gridSize), random.randint(0, gridSize))

            while ((top_right[0] <= bottom_left[0]) or (top_right[1] <= bottom_left[1])):
                bottom_left = (random.randint(0, gridSize), random.randint(0, gridSize))
                top_right = (random.randint(0, gridSize), random.randint(0, gridSize))
    
            width = top_right[0] - bottom_left[0]
            height = top_right[1] - bottom_left[1]
            Area = width * height

            if (Area < maxArea):
                overlap = checkOverlap(bottom_left, top_right, Rectangles)
            else:
                overlap = True

        Rectangles.append(Rectangle(i, bottom_left[0], bottom_left[1], top_right[0], top_right[1]))

    return Rectangles

def genAvgCaseRects(numrect, gridSize, maxArea):
    Rectangles = []
    side_length = math.sqrt(maxArea)
    width = side_length
    height = side_length
    x = 0
    y = 0

    for i in range(numrect):
        bottom_left = (x, y)
        top_right = (x + width, y + height)
        Rectangles.append(Rectangle(i, bottom_left[0], bottom_left[1], top_right[0], top_right[1]))

        x += width  # Move to the right for the next rectangle
        if x + width > gridSize:  # If the next rectangle exceeds the grid's width
            x = 0  # Move to the leftmost position in the next row
            y += height  # Move to the next row

    return Rectangles



def genWorstCaseRects(numrects, gridSize, maxArea):
    Rectangles = []
    width = gridSize / numrects
    height = maxArea / width

    for i in range(numrects):
        bottom_left = (i * width, 0)
        top_right = ((i + 1) * width, height)
        Rectangles.append(Rectangle(i, bottom_left[0], bottom_left[1], top_right[0], top_right[1]))

    return Rectangles

def genBestCaseRects(numrects, gridSize, maxArea):
    Rectangles = []
    height = gridSize / numrects
    width = maxArea / height

    for i in range(numrects):
        bottom_left = (0, i * height)
        top_right = (width, (i + 1) * height)
        Rectangles.append(Rectangle(i, bottom_left[0], bottom_left[1], top_right[0], top_right[1]))

    return Rectangles




def visualise(Rectangles, area):
    # visualiing the rectangles
    fig, ax = plt.subplots()
    for rect in Rectangles:
        x1, y1 = rect.bottom_left
        x2, y2 = rect.top_right
        width = x2 - x1
        height = y2 - y1
        ax.add_patch(plt.Rectangle((x1, y1), width, height, facecolor='white', edgecolor='blue'))
        label_x = x1 + width / 2
        label_y = y1 + height / 2
        ax.annotate(rect.rectNum, (label_x, label_y), color='blue', ha='center', va='center', fontsize=5)

    ax.set_xlim(-1,  area + 1)
    ax.set_ylim(-1,  area + 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_facecolor('grey')
    plt.show()

def plotStats(n_list, bf_times, o_times):
    # Plot the data and curve fitting
    fig, ax = plt.subplots()
    ax.plot(n_list, bf_times, label='BF data')
    ax.plot(n_list, o_times, label='Opt data')
    ax.set_xlabel('n')
    ax.set_ylabel('time (ms)')

    # Curve fitting for BF data
    pfit_bf = np.polyfit(n_list, bf_times, 2)
    yfit_bf = np.polyval(pfit_bf, n_list)
    ax.plot(n_list, yfit_bf, label='BF curve fitting (degree 2)')

    # Curve fitting for Opt data
    pfit_opt = np.polyfit(n_list, o_times, 2)
    yfit_opt = np.polyval(pfit_opt, n_list)
    ax.plot(n_list, yfit_opt, label='Opt curve fitting (degree 2)')

    # Statistical data for BF data
    coefficients_bf = pfit_bf
    p_values_bf = np.polyfit(n_list, bf_times, 2)
    conf_int_bf = 1.96 * np.std(bf_times) / np.sqrt(len(bf_times))
    conf_level = 95
    conf_percent = round((1 - conf_level/100) * 100, 2)
    textstr_bf = '\n'.join((
        r'BF Regression line: $y = %.3fx^2 + %.3fx + %.3f$' % (coefficients_bf[2], coefficients_bf[1], coefficients_bf[0]),
        r'BF R-squared: %.3f' % (np.corrcoef(n_list, bf_times)[0, 1] ** 2),
        r'BF p-values: $p_1=%.3f$, $p_2=%.3f$, $p_3=%.3f$' % (p_values_bf[2], p_values_bf[1], p_values_bf[0]),
        r'BF Confidence intervals: $[%.3f, %.3f]$, $[%.3f, %.3f]$, $[%.3f, %.3f]$' % (
            coefficients_bf[2] - conf_int_bf, coefficients_bf[2] + conf_int_bf,
            coefficients_bf[1] - conf_int_bf, coefficients_bf[1] + conf_int_bf,
            coefficients_bf[0] - conf_int_bf, coefficients_bf[0] + conf_int_bf),
        r'Confidence: %d%%' % conf_level))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.85, textstr_bf, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    # Statistical data for Opt data
    coefficients_opt = pfit_opt
    p_values_opt = np.polyfit(n_list, o_times, 2)
    conf_int_opt = 1.96 * np.std(o_times) / np.sqrt(len(o_times))
    textstr_opt = '\n'.join((
        r'Opt Regression line: $y = %.3fx^2 + %.3fx + %.3f$' % (coefficients_opt[2], coefficients_opt[1], coefficients_opt[0]),
        r'Opt R-squared: %.3f' % (np.corrcoef(n_list, o_times)[0, 1] ** 2),
        r'Opt p-values: $p_1=%.3f$, $p_2=%.3f$, $p_3=%.3f$' % (p_values_opt[2], p_values_opt[1], p_values_opt[0]),
        r'Opt Confidence intervals: $[%.3f, %.3f]$, $[%.3f, %.3f]$, $[%.3f, %.3f]$' % (
            coefficients_opt[2] - conf_int_opt, coefficients_opt[2] + conf_int_opt,
            coefficients_opt[1] - conf_int_opt, coefficients_opt[1] + conf_int_opt,
            coefficients_opt[0] - conf_int_opt, coefficients_opt[0] + conf_int_opt),
        r'Confidence: %d%%' % conf_level))

    ax.text(0.05, 0.70, textstr_opt, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower right')
    plt.show()

    
    # # Create the table
    # fig, ax = plt.subplots()
    # table_data = [['n', 'time (ms)']]
    # for i in range(len(n_list)):
    #     table_data.append([n_list[i], bf_times[i]])
    # table_data = tuple(table_data)
    # table = ax.table(cellText=table_data, loc='center')
    # table.auto_set_font_size(False)
    # table.set_fontsize(12)
    # table.scale(1, 1.5)
    # ax.axis('off')
    # plt.show()

def plotStatsCases(n_list, best_times, worst_times, avg_times, random_times):
    # Plot the data and curve fitting
    fig, ax = plt.subplots()

    # Plotting the data
    ax.plot(n_list, best_times, label='Best Case')
    ax.plot(n_list, worst_times, label='Worst Case')
    ax.plot(n_list, avg_times, label='Average Case')
    ax.plot(n_list, random_times, label='Random Case')

    ax.set_xlabel('n')
    ax.set_ylabel('time (ms)')

    # Curve fitting for Best Case
    pfit_best = np.polyfit(n_list, best_times, 2)
    yfit_best = np.polyval(pfit_best, n_list)
    ax.plot(n_list, yfit_best, label='Best Case Curve Fitting (degree 2)')

    # Curve fitting for Worst Case
    pfit_worst = np.polyfit(n_list, worst_times, 2)
    yfit_worst = np.polyval(pfit_worst, n_list)
    ax.plot(n_list, yfit_worst, label='Worst Case Curve Fitting (degree 2)')

    # Curve fitting for Average Case
    pfit_avg = np.polyfit(n_list, avg_times, 2)
    yfit_avg = np.polyval(pfit_avg, n_list)
    ax.plot(n_list, yfit_avg, label='Average Case Curve Fitting (degree 2)')

    # Curve fitting for Random Case
    pfit_random = np.polyfit(n_list, random_times, 2)
    yfit_random = np.polyval(pfit_random, n_list)
    ax.plot(n_list, yfit_random, label='Random Case Curve Fitting (degree 2)')

    # Statistical data for Best Case
    coefficients_best = pfit_best
    p_values_best = np.polyfit(n_list, best_times, 2)
    conf_int_best = 1.96 * np.std(best_times) / np.sqrt(len(best_times))
    textstr_best = '\n'.join((
        r'Best Case Regression line: $y = %.3fx^2 + %.3fx + %.3f$' % (coefficients_best[2], coefficients_best[1], coefficients_best[0]),
        r'Best Case R-squared: %.3f' % (np.corrcoef(n_list, best_times)[0, 1] ** 2),
        r'Best Case p-values: $p_1=%.3f$, $p_2=%.3f$, $p_3=%.3f$' % (p_values_best[2], p_values_best[1], p_values_best[0]),
        r'Best Case Confidence intervals: $[%.3f, %.3f]$, $[%.3f, %.3f]$, $[%.3f, %.3f]$' % (
            coefficients_best[2] - conf_int_best, coefficients_best[2] + conf_int_best,
            coefficients_best[1] - conf_int_best, coefficients_best[1] + conf_int_best,
            coefficients_best[0] - conf_int_best, coefficients_best[0] + conf_int_best)))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.85, textstr_best, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    # Statistical data for Worst Case
    coefficients_worst = pfit_worst
    p_values_worst = np.polyfit(n_list, worst_times, 2)
    conf_int_worst = 1.96 * np.std(worst_times) / np.sqrt(len(worst_times))
    textstr_worst = '\n'.join((
        r'Worst Case Regression line: $y = %.3fx^2 + %.3fx + %.3f$' % (coefficients_worst[2], coefficients_worst[1], coefficients_worst[0]),
        r'Worst Case R-squared: %.3f' % (np.corrcoef(n_list, worst_times)[0, 1] ** 2),
        r'Worst Case p-values: $p_1=%.3f$, $p_2=%.3f$, $p_3=%.3f$' % (p_values_worst[2], p_values_worst[1], p_values_worst[0]),
        r'Worst Case Confidence intervals: $[%.3f, %.3f]$, $[%.3f, %.3f]$, $[%.3f, %.3f]$' % (
            coefficients_worst[2] - conf_int_worst, coefficients_worst[2] + conf_int_worst,
            coefficients_worst[1] - conf_int_worst, coefficients_worst[1] + conf_int_worst,
            coefficients_worst[0] - conf_int_worst, coefficients_worst[0] + conf_int_worst)))

    ax.text(0.05, 0.70, textstr_worst, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    # Statistical data for Average Case
    coefficients_avg = pfit_avg
    p_values_avg = np.polyfit(n_list, avg_times, 2)
    conf_int_avg = 1.96 * np.std(avg_times) / np.sqrt(len(avg_times))
    textstr_avg = '\n'.join((
        r'Average Case Regression line: $y = %.3fx^2 + %.3fx + %.3f$' % (coefficients_avg[2], coefficients_avg[1], coefficients_avg[0]),
        r'Average Case R-squared: %.3f' % (np.corrcoef(n_list, avg_times)[0, 1] ** 2),
        r'Average Case p-values: $p_1=%.3f$, $p_2=%.3f$, $p_3=%.3f$' % (p_values_avg[2], p_values_avg[1], p_values_avg[0]),
        r'Average Case Confidence intervals: $[%.3f, %.3f]$, $[%.3f, %.3f]$, $[%.3f, %.3f]$' % (
            coefficients_avg[2] - conf_int_avg, coefficients_avg[2] + conf_int_avg,
            coefficients_avg[1] - conf_int_avg, coefficients_avg[1] + conf_int_avg,
            coefficients_avg[0] - conf_int_avg, coefficients_avg[0] + conf_int_avg)))

    ax.text(0.05, 0.55, textstr_avg, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    # Statistical data for Random Case
    coefficients_random = pfit_random
    p_values_random = np.polyfit(n_list, random_times, 2)
    conf_int_random = 1.96 * np.std(random_times) / np.sqrt(len(random_times))
    textstr_random = '\n'.join((
        r'Random Case Regression line: $y = %.3fx^2 + %.3fx + %.3f$' % (coefficients_random[2], coefficients_random[1], coefficients_random[0]),
        r'Random Case R-squared: %.3f' % (np.corrcoef(n_list, random_times)[0, 1] ** 2),
        r'Random Case p-values: $p_1=%.3f$, $p_2=%.3f$, $p_3=%.3f$' % (p_values_random[2], p_values_random[1], p_values_random[0]),
        r'Random Case Confidence intervals: $[%.3f, %.3f]$, $[%.3f, %.3f]$, $[%.3f, %.3f]$' % (
            coefficients_random[2] - conf_int_random, coefficients_random[2] + conf_int_random,
            coefficients_random[1] - conf_int_random, coefficients_random[1] + conf_int_random,
            coefficients_random[0] - conf_int_random, coefficients_random[0] + conf_int_random)))

    ax.text(0.05, 0.40, textstr_random, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower right')
    plt.show()


   # Table for execution times
    fig, ax_table = plt.subplots(figsize=(4, 2))
    ax_table.axis('off')

    table_data = [
        ['Case', 'Average Time (ms)'],
        ['Best Case', f'{np.mean(best_times):.5e}'],
        ['Worst Case', f'{np.mean(worst_times):.5e}'],
        ['Average Case', f'{np.mean(avg_times):.5e}'],
        ['Random Case', f'{np.mean(random_times):.5e}']
    ]
    table = ax_table.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.tight_layout()
    plt.show()



def compareAdjacencyLists(rectangles1, rectangles2):
    if len(rectangles1) != len(rectangles2):
        return False

    for rect1, rect2 in zip(rectangles1, rectangles2):
        if len(rect1.adjacencies) != len(rect2.adjacencies):
            return False
        
        # Check if each rectangle in rect1.adjacencies exists in rect2.adjacencies
        for adj_rect1 in rect1.adjacencies:
            found_match = False
            for adj_rect2 in rect2.adjacencies:
                if adj_rect1.equals(adj_rect2):
                    found_match = True
                    break
            if not found_match:
                return False

    return True



def checkAdjacenciesBruteForce(Rectangles):
    for r1 in Rectangles:
        for r2 in Rectangles:
            if not (r1.equals(r2)):
                if ((r1.top_right[0] == r2.bottom_left[0]) and 
                (((r1.bottom_left[1] == r2.bottom_left[1]) and (r1.top_right[1] == r2.top_right[1])) or
                ((r1.bottom_left[1] > r2.bottom_left[1]) and r1.bottom_left[1] < r2.top_right[1]) or 
                ((r1.top_right[1] > r2.bottom_left[1]) and r1.top_right[1] < r2.top_right[1]) or
                ((r1.bottom_left[1] <= r2.bottom_left[1]) and r1.top_right[1] >= r2.top_right[1]))):
                    r1.adjacencies.append(r2)

    return Rectangles


def checkAdjacenciesOptimized(Rectangles):
    events = []
    for r in Rectangles:
        events.append((r.bottom_left[1], "start", r))
        events.append((r.top_right[1], "end", r))
    events.sort(key=lambda x: (x[0], x[1]))

    active = SortedList(key=lambda r: r.bottom_left[1])
    
    for event in events:
        if event[1] == "start":
            for r in active:
                if r.top_right[0] == event[2].bottom_left[0]:
                    r.adjacencies.append(event[2])
                elif event[2].top_right[0] == r.bottom_left[0]:
                    event[2].adjacencies.append(r)
            active.add(event[2])
        elif event[1] == "end":
            active.remove(event[2])
    
    return Rectangles



# Making an array to test the brute force vs optimised algo over different sample sizes
def testBFvsOpt(n_vlaues, upLimit, gridSize, Area):
    n_list = []
    for i in range(n_vlaues):
        value = int(10 + (i / 99) * (upLimit- 10))
        n_list.append(value)

    bf_times = []
    o_times= [] 

    for n in n_list:
        Rectangles = genRects(n, gridSize, Area)
        bf_Rectangles  = copy.deepcopy(Rectangles) 
        o_Rectangles  = copy.deepcopy(Rectangles)

        bf_start_time = time.time()
        bf_Rectangles = checkAdjacenciesBruteForce(bf_Rectangles)
        bf_end_time = time.time()
        bf_elapsed_time = (bf_end_time - bf_start_time) * 10**3
        bf_times.append(bf_elapsed_time)

        o_start_time = time.time()
        o_Rectangles = checkAdjacenciesOptimized(o_Rectangles)
        o_end_time = time.time()
        o_elapsed_time = (o_end_time - o_start_time) * 10**3
        o_times.append(o_elapsed_time)

    return n_list, bf_times, o_times

def testCases(n_vlaues, upLimit, gridSize, Area):
    n_list = []
    for i in range(50):
        value = int(10 + (i / 99) * (4000- 10))
        n_list.append(value)

    b_times = []
    w_times= [] 
    a_times = []
    r_times = []

    for n in n_list:
        b_Rectangles = genBestCaseRects(n, 200, 50)
        w_Rectangles = genWorstCaseRects(n, 200, 50)
        a_Rectangles = genAvgCaseRects(n, 200, 50)
        r_Rectangles = genRects(n, 200, 50)


        b_start_time = time.time()
        b_Rectangles = checkAdjacenciesOptimized(b_Rectangles)
        b_end_time = time.time()
        b_elapsed_time = (b_end_time - b_start_time) * 10**3
        b_times.append(b_elapsed_time)

        w_start_time = time.time()
        w_Rectangles = checkAdjacenciesOptimized(w_Rectangles)
        w_end_time = time.time()
        w_elapsed_time = (w_end_time - w_start_time) * 10**3
        w_times.append(w_elapsed_time)

        a_start_time = time.time()
        a_Rectangles = checkAdjacenciesOptimized(a_Rectangles)
        a_end_time = time.time()
        a_elapsed_time = (a_end_time - a_start_time) * 10**3
        a_times.append(a_elapsed_time)

        r_start_time = time.time()
        r_Rectangles = checkAdjacenciesOptimized(r_Rectangles)
        r_end_time = time.time()
        r_elapsed_time = (r_end_time - r_start_time) * 10**3
        r_times.append(r_elapsed_time)

    return n_list, b_times, w_times, a_times, r_times


#Example of test

n_list, bf_times, o_times = testBFvsOpt(50, 2000, 200, 50)

plotStats(n_list, bf_times, o_times)























