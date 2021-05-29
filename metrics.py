#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 15:57:44 2021

@author: mason
"""
def generate_metrics():
    for i in range(0, len(ticker), 1):
        name = ticker[i]
        x = actual_close_prices[name]
        y = metrics_dict[name]
        deviation = ((abs(x - y)) / x) * 100
        deviation = round(deviation, 2)
        deviation_dict[name] = deviation
        deviation = str(deviation)
        print('The deviation for ' + name + ' is: ' + deviation + '%.')
    return

def average_deviation():
    a = 0
    for i in range(0, len(ticker), 1):
        name = ticker[i]
        a = a + deviation_dict[name]
    a = a/len(ticker)
    a = round(a, 2)
    print()
    print('The average deviation between all ' + str(len(ticker)) + ' stocks is: ' + str(a) + '.')
    
    
    

