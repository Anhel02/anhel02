import numpy as np
import matplotlib.pyplot as plt

def lab_round(numbers, uncertainties):
    '''
    Laboratory Convention Rounding:

    1. If the uncertainty being dealt with contains only one Significant Figure (SF) (a common case in direct measurements where uncertainty arises solely from precision issues), it will be left as obtained, with just one SF.
    2. Uncertainties with two or more SFs are rounded to just 2 SFs.
    3. The result is rounded to the same number of decimal places as the uncertainty.

    '''

    # Lists to store the rounded values:
    rounded_numbers = []
    rounded_uncertainties = []
    
    for number, uncertainty in zip(numbers, uncertainties):
        # Contar la cantidad de SF tanto de los valores como de las incertidumbres
        num_SF = len(str(abs(number)).rstrip('0').rstrip('.'))
        uncertainty_SF = len(str(abs(uncertainty)).rstrip('0').rstrip('.'))
        
        # Counting the number of Significant Figures (SFs) in both the values and the uncertainties.
        decimal_places = min(num_SF, uncertainty_SF)
        
        # Redondear el número y la incertidumbre
        rounded_number = round(number, decimal_places)
        rounded_uncertainty = round(uncertainty, decimal_places)
        
        # Rounding the number and the uncertainty.
        rounded_number_str = str(rounded_number).rstrip('0').rstrip('.') if '.' in str(rounded_number) else str(rounded_number)
        
        # Ensuring that the numbers have the same number of digits as the uncertainties.
        if len(rounded_number_str) < len(str(rounded_uncertainty)):
            rounded_number_str += '0' * (len(str(rounded_uncertainty)) - len(rounded_number_str))
            # If two consecutive zeros appear, refrain from adding more zeros.
            if rounded_number_str[-2:] == '00':
                break

        
        # Adding the rounded values to the created lists.
        rounded_numbers.append(rounded_number_str)
        rounded_uncertainties.append(rounded_uncertainty)
    
    return rounded_numbers, rounded_uncertainties

class lin_reg:
    '''
    Linear Regression.
    '''
    
    def __init__(self,x,delta_x,y,delta_y):
        self.x = np.array(x)
        self.delta_x = np.array(delta_x)
        self.y = np.array(y)
        self.delta_y = np.array(delta_y)
    
    def lin_coef(self):
        '''
        Obtaining the coefficients m and n along with their errors.
        '''
        N = len(self.x)
        m = (N * np.sum (self.x*self.y) - np.sum(self.x) * np.sum(self.y))/(N*np.sum(self.x**2) - (np.sum(self.x))**2)
        n = (np.sum(self.x**2) * np.sum(self.y) - np.sum(self.x) * np.sum(self.x * self.y))/(N * np.sum(self.x**2)-(np.sum(self.x))**2)
        delta_m = np.sqrt((N * np.sum(self.y-n-m*self.x)**2)/(N - 2) * (N * np.sum(self.x**2) - (np.sum(self.x))**2))
        delta_n = np.sqrt((np.sum(self.x**2) * np.sum(self.y-n-m*self.x)**2)/((N - 2) * (N * np.sum(self.x**2) - (np.sum(self.x))**2)))
        return m,delta_m,n,delta_n

    def datafit(self):
        '''
        Generating an array of points for the fitted regression line.
        '''
        x_fit = np.linspace(min(self.x),max(self.x),100)
        delta_x_fit = (max(self.x)-min(self.x))/(2*100)
        y_fit = np.array([])
        delta_y_fit = np.array([])
        m, delta_m, n, delta_n = self.lin_coef()

        for i in range(len(x_fit)):
            y_fit = np.append(y_fit,m*x_fit[i]+n)
            delta_y_fit = np.append(delta_y_fit, np.sqrt((delta_m*x_fit[i])**2 + delta_n + (m*self.delta_x)**2 + self.delta_y**2))
        return x_fit,delta_x_fit,y_fit,delta_y_fit
        
    def r2(self):
        '''
        Calculating the correlation coefficient for linear regressions.
        '''
        N = len(self.x)
        r2 = (N * np.sum(self.x * self.y) - np.sum(self.x) * np.sum(self.y))/(np.sqrt((N * np.sum(self.x**2) - (np.sum(self.x))**2) * (N * np.sum(self.y**2) - (np.sum(self.y))**2)))
        return r2
