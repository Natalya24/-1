
#табельный номер tabel_number
#ФИО name
#пол gender
#год рождения year_of_birth
#год начала работы в компании year_of_start_working
#подразделение subdivision
#должность job_title
#оклад salary
#количество выполненных проектов count_of_projects

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from random import shuffle
from russian_names import RussianNames
import csv

#генерирование и сохранение набора данных в виде csv
filename = r'dataset.csv'

russiannames_men = RussianNames(count = random.randrange(500, 1000), gender = 1.0, patronymic = True, name_reduction = True, patronymic_reduction = True, transliterate = False)
russiannames_women = RussianNames(count = random.randrange(500, 1000), gender = 0.0, patronymic = True, name_reduction = True, patronymic_reduction = True, transliterate = False)

gender_list = list()
for i in range(len(russiannames_men)):
    gender_list.append('мужской')
for i in range(len(russiannames_women)):
    gender_list.append('женский')

tabel_number = list()
for i in range(len(gender_list)):
    tabel_number.append(i)

year_of_birth_list = list()
for i in range(len(russiannames_men)):
    year_of_birth_list.append(int(random.randrange(1960,1998)))
for i in range(len(russiannames_women)):
    year_of_birth_list.append(int(random.randrange(1965,1998)))

year_of_start_list = list()
for i in range(len(russiannames_men)):
    year_of_start_list.append(int(random.randrange(2010,2022)))
for i in range(len(russiannames_women)):
    year_of_start_list.append(int(random.randrange(2010,2022)))

subdivision_list = list()
subdivisions = ['Отдел маркетинга', 'Отдел продаж', 'Отдел закупок', 'Сервис']
for i in range(len(gender_list)):
    for j in subdivisions:
        subdivision_list.append(j)
shuffle(subdivision_list)

job_title_list = list()
job_titles = ["Руководитель", 'Стажер', 'Младший сотрудник', 'Специалист', 'Старший сотрудник']
for i in range(len(gender_list)):
    job_title_list.append(job_titles[random.randrange(0,4)])

salary_list = list()
count_of_projects_list = list()
for item in job_title_list:
    if item == 'Руководитель':
        salary_list.append(float(random.randrange(100000, 200000)))
        count_of_projects_list.append(int(random.randrange(40,50)))
    if item == 'Стажер':
        salary_list.append(float(random.randrange(10000, 20000)))
        count_of_projects_list.append(int(random.randrange(5)))
    if item == 'Младший сотрудник':
        salary_list.append(float(random.randrange(20000, 30000)))
        count_of_projects_list.append(int(random.randrange(5,10)))
    if item == 'Специалист':
        salary_list.append(float(random.randrange(30000, 50000)))
        count_of_projects_list.append(int(random.randrange(10,30)))
    if item == 'Старший сотрудник':
        salary_list.append(float(random.randrange(50000, 80000)))
        count_of_projects_list.append(int(random.randrange(30,40)))

name_list = list()
for item in russiannames_men:
    name_list.append(item)
for item in russiannames_women:
    name_list.append(item)

people = list(zip(tabel_number, name_list, gender_list, year_of_birth_list, year_of_start_list, subdivision_list, job_title_list, salary_list, count_of_projects_list))
shuffle(people)

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(people)

#прочитать сгенерированный набор данных в виде списков
#получить для разных по типу признаков столбцов основные статистические характеристики
#минимум, максимум, среднее, дисперсия, стандартное отклонение, медиана
with open(filename, 'r', newline='') as file:
    reader = csv.reader(file)
    list_of_csv = list(reader)

print("Основные статистические характеристики - numpy")
print('Количество сотрудников: ' + str(len(list_of_csv)) + ' человек')
salary = list()
for row in list_of_csv:
    salary.append(float(row[7]))
year_of_birth = list()
for row in list_of_csv:
    year_of_birth.append(2022-int(row[3]))
count_of_projects = list()
for row in list_of_csv:
    count_of_projects.append(int(row[8]))
print('Максимум: максимальный оклад - ' + str(np.max(salary)) + ' самый старший возраст работника - ' + str(np.max(year_of_birth)) + ' самое большое количество выполненных проектов у одного сотрудника - ' + str(np.max(count_of_projects)))
print('Минимум: минимальный оклад - ' + str(np.min(salary)) + ' самый младший возраст работника - ' + str(np.min(year_of_birth)) + ' самое минимальное количество выполненных проектов у одного сотрудника - ' + str(np.min(count_of_projects)))
print('Среднее: средний оклад - ' + str(round(np.mean(salary), 3)) + ' средний возраст работника - ' + str(round(np.mean(year_of_birth))) + ' среднее количество выполненных проектов у одного сотрудника - ' + str(round(np.mean(count_of_projects))))
print('Дисперсия: по окладу - ' + str(round(np.var(salary), 3)) + ' по возрасту - ' + str(round(np.var(year_of_birth))) + ' по проектам - ' + str(round(np.var(count_of_projects))))
print('Медиана: по окладу - ' + str(round(np.median(salary), 3)) + ' по возрасту - ' + str(round(np.median(year_of_birth))) + ' по проектам - ' + str(round(np.median(count_of_projects))))

#прочитать сгенерированный набор данных в виде датафрейма
#получить для разных по типу признаков столбцов основные статистические характеристики
#минимум, максимум, среднее, дисперсия, стандартное отклонение, медиана
df = pd.read_csv(filename, delimiter = ',', names = ['Табельный номер', 'ФИО', 'пол', 'год рождения', 'год начала работы', 'подразделение', 'должность', 'оклад', 'количество проектов'])
print("---------------------")
print("Основные статистические характеристики - pandas")
print('Максимум: максимальный оклад - ' + str(df['оклад'].max()) + ' самый старший возраст работника - ' + str(df['год рождения'].max()) + ' самое большое количество выполненных проектов у одного сотрудника - ' + str(df['количество проектов'].max()))
print('Минимум: минимальный оклад - ' + str(df['оклад'].min()) + ' самый младший возраст работника - ' + str(df['год рождения'].min()) + ' самое минимальное количество выполненных проектов у одного сотрудника - ' + str(df['количество проектов'].min()))
print('Среднее: средний оклад - ' + str(round(df['оклад'].mean(), 3)) + ' средний возраст работника - ' + str(round(df['год рождения'].mean())) + ' среднее количество выполненных проектов у одного сотрудника - ' + str(round(df['количество проектов'].mean())))
print('Дисперсия: по окладу - ' + str(round(df['оклад'].var(), 3)) + ' по возрасту - ' + str(round(df['год рождения'].var())) + ' по проектам - ' + str(round(df['количество проектов'].var())))
print('Медиана: по окладу - ' + str(round(df['оклад'].median(), 3)) + ' по возрасту - ' + str(round(df['год рождения'].median())) + ' по проектам - ' + str(round(df['количество проектов'].median())))

#Построить не менее трех разнотипных графиков
x = list()
for i in range(50):
    x.append(tabel_number[i])

y = list()
for i in range(50):
    y.append(count_of_projects_list[i])

plt.bar(x, y, color = 'g', width = 0.5, label = "Кол-во проектов")
plt.xlabel('Табельный номер')
plt.ylabel('Кол-во проектов')
plt.title('План выполнения проектов')
plt.legend()
plt.show()

z = list()
for i in range(50):
    z.append(salary [i])

plt.plot(x, z,  color = 'g', linestyle = 'dashed', marker = 'o',label = "Уровень заработной платы")
plt.xticks(rotation = 25)
plt.xlabel('Табельный номер')
plt.ylabel('Оклад')
plt.title('Уровень заработной платы', fontsize = 20)
plt.grid()
plt.legend()
plt.show()

j = list()
for i in range(50):
    j.append(2022 - (year_of_birth_list[i]))

plt.scatter(x, j, color = 'g',s = 10)
plt.xticks(rotation = 25)
plt.xlabel('Табельный номер')
plt.ylabel('Год рождения')
plt.title('Распределение возрастов', fontsize = 20)
plt.show()
