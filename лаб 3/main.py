
import csv
from random import shuffle

from russian_names import RussianNames

#exercise_1
fieldnames = [
  'Табельный номер', 'Фамилия И. О.', 'Пол', 'Год рождения', 'Начало работы (год)', 'Подразделение', 'Должность', 'Оклад', 'Кол-во проектов'
]

def lastname_io ():
  lastnames = []
  row_mens = RussianNames(count=500, gender=1.0, patronymic=True, 
    name_reduction=True, patronymic_reduction=True)
  row_woman = RussianNames(count=500, gender=0.0, patronymic=True, 
    name_reduction=True, patronymic_reduction=True)
  for i in row_mens:
    lastnames.append([i])

  for j in row_woman:
    lastnames.append([j])
  return lastnames

def gender():
  gender = [['М']] * 500 + [['Ж']] * 500
  return gender

def year_of_birth():
  years = []
  for i in range(1, 1001):
    for j in range(1973, 2007):
      years.append([j])
  return years

def start_work():
  years = []
  for i in range(1, 1001):
    for j in range(2005, 2022):
      years.append([j])
  return years

def money():
  money = []
  for i in range(1, 1001):
    for j in range(45000, 48500):
      money.append([j])
  return money

def count():
  count = []
  for i in range(1, 1001):
    for j in range(2, 10):
      count.append([j])
  return count

def staff(): 
  staff = []
  info = ['Департамент международного сотрудничества',
          'Производственно-технический комплекс',
          'Финансово-экономический отдел',
          'Служба безопасности',
          'Режимно-секретный отдел',
          'Департамент управления закупками',
          'Бухгалтерия',
          'Служба управления персоналом']
  for i in range(1, 1001):
    for j in info:
      staff.append([j])
  return staff

def post(): 
  post = []
  info = ['Менеджер по продажам',
          'Инженер',
          'Экономист',
          'Специалист', 
          'Начальник отдела',
          'Бугалтер',
          'Старший специалист']
  for i in range(1, 1001):
    for j in info:
      post.append([j])
  return post

#exercise_2
import numpy
import pandas as pnd

file = 'data.csv'
with open(file, 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerow([g for g in fieldnames])

  post = post()
  staff = staff()
  count = count()
  mon = money()
  years = year_of_birth()
  work = start_work()
  lastname = lastname_io()
  gender = gender()
  
  a = list(zip(lastname, gender, years, work, staff, post, mon, count))
  shuffle(a)
  
  for i,(j,z,b,w,staff,post,mon,count) in enumerate(a, 1):
    writer.writerow([i] + j+z+b+w+staff+post+mon+count)

pnd.set_option('display.max_rows', None)
pnd.set_option('display.max_columns', None)
pnd.set_option('display.max_colwidth', None)

df = pnd.read_csv(file, sep=',')
print('---------------- Основная информация ----------------')
print('Количество сотрудников: ' + str(df['Табельный номер'].count()) + ' человек')
print('Максимальный оклад: ' + str(df['Оклад'].max()) + ' рублей')
print('Минимальный оклад: ' + str(df['Оклад'].min()) + ' рублей')
print('Бюджет компании на ЗП: ' + str(df['Оклад'].sum()) + ' рублей')
print('Среднее кол-во проектов на 1го человека: ' + str(df['Кол-во проектов'].mean()) + ' проекта' )
print('Медиана возраста: ' + str(df['Год рождения'].median()) + ' год\n')
print(df[['Табельный номер', 'Год рождения', 'Оклад', 'Кол-во проектов']].describe())

#exercise_3
import matplotlib.pyplot as plt

x = []
y = []

with open('data.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')

    for row in plots:
        x.append(row[0])
        y.append(row[8])
  
plt.bar(x, y, color = 'g', width = 0.5, label = "Кол-во проектов")
plt.xlabel('Табельный номер')
plt.ylabel('Кол-во проектов')
plt.title('План выполнения проектов')
plt.legend()
plt.show()

with open('data.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        x.append(row[0])
        y.append(row[7])
  
plt.plot(x, y, color = 'g', linestyle = 'dashed',
         marker = 'o',label = "Уровень заработной платы")
  
plt.xticks(rotation = 25)
plt.xlabel('Dates')
plt.ylabel('Оклад')
plt.title('Уровень заработной платы', fontsize = 20)
plt.grid()
plt.legend()
plt.show()

with open('data.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        x.append(row[0])
        y.append(int(row[1]))
  
plt.scatter(x, y, color = 'g',s = 100)
plt.xticks(rotation = 25)
plt.xlabel('Табельный номер')
plt.ylabel('Оклад')
plt.title('Динамика заработной платы', fontsize = 20)
plt.show()
