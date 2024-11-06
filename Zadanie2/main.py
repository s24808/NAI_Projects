# AUTHORS: Filip Labuda, Jędrzej Stańczewski

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Definicja zmiennych wejściowych z odpowiednimi zakresami

# light_level: natężenie światła (0-100), gdzie 0 oznacza
# nieskie natężenie światła, a 100 największe natężenie światła

# time_of_day: pora dnia (0-24), gdzie 0 oznacza północ, 12 oznacza południe,
# a 24 również północ kończąc przy tym cykl pełnego dnia

# occupancy: stan zajętości pomieszczenia (0 i 1), gdzie 0 oznacza brak osób w
# pomieszczeniu, a 1 oznacza, że pomieszczenie jest zajęte

# light_intensity: zakres intensywności oświetlenie (0-100), gdzie 0 oznacza brak światła,
# a 100 maksymalną intensywnosć oświetlenie

light_level = ctrl.Antecedent(np.arange(0, 101, 1), 'light_level')
time_of_day = ctrl.Antecedent(np.arange(0, 25, 1), 'time_of_day')
occupancy = ctrl.Antecedent(np.arange(0, 2, 1), 'occupancy')

# Definicja zmiennej wyjściowej
light_intensity = ctrl.Consequent(np.arange(0, 101, 1), 'light_intensity')

# Zakres wartości dla light_level
light_level['low'] = fuzz.trimf(light_level.universe, [0, 0, 50])
light_level['medium'] = fuzz.trimf(light_level.universe, [25, 50, 75])
light_level['high'] = fuzz.trimf(light_level.universe, [50, 100, 100])

# Zakres wartości dla time_of_day
time_of_day['morning'] = fuzz.trimf(time_of_day.universe, [6, 9, 12])
time_of_day['day'] = fuzz.trimf(time_of_day.universe, [10, 14, 18])
time_of_day['evening'] = fuzz.trimf(time_of_day.universe, [16, 19, 22])
time_of_day['night'] = fuzz.trimf(time_of_day.universe, [20, 24, 24])

# Zakres wartości dla occupancy
occupancy['empty'] = fuzz.trimf(occupancy.universe, [0, 0, 1])
occupancy['full'] = fuzz.trimf(occupancy.universe, [0, 1, 1])

# Zakres wartości dla light_intensity
light_intensity['low'] = fuzz.trimf(light_intensity.universe, [0, 0, 50])
light_intensity['medium'] = fuzz.trimf(light_intensity.universe, [25, 50, 75])
light_intensity['high'] = fuzz.trimf(light_intensity.universe, [50, 100, 100])

# Definicja zasad rozmytych (przykładowe możliwe scenariusze)
rule1 = ctrl.Rule(light_level['high'] & time_of_day['day'] & occupancy['full'], light_intensity['low'])
rule2 = ctrl.Rule(light_level['low'] & time_of_day['night'], light_intensity['high'])
rule3 = ctrl.Rule(light_level['medium'] & time_of_day['evening'] & occupancy['full'], light_intensity['medium'])
rule4 = ctrl.Rule(light_level['low'] & time_of_day['morning'] & occupancy['full'], light_intensity['medium'])
rule5 = ctrl.Rule(light_level['high'] & time_of_day['night'], light_intensity['low'])
rule6 = ctrl.Rule(light_level['medium'] & time_of_day['day'], light_intensity['medium'])

# System kontroli na posdtawie zasad
lighting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
lighting_test = ctrl.ControlSystemSimulation(lighting_control)

# Przykładowe wartości wejściowe
# Niskie natężenie światła
lighting_test.input['light_level'] = 30
# Wieczór
lighting_test.input['time_of_day'] = 21
# Pomieszczenie zajęte
lighting_test.input['occupancy'] = 1

# Sprawdzenie działania aplikacji
lighting_test.compute()

# Wyświetlenie wyniku
print(f"Dane wejściowe: ")
print(lighting_test.input)

print(f"Wynik: ")
print(f"Light Intensity: {lighting_test.output['light_intensity']}")
light_intensity.view(sim=lighting_test)
