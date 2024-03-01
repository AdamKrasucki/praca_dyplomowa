# Databricks notebook source
# MAGIC %md
# MAGIC ###1. Import danych

# COMMAND ----------

pip install pmdarima # dodatkowa biblioteka do automatycznego doboru parametrów modelu SARIMA

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # bibliotka do wykresów
from statsmodels.tsa.statespace.sarimax import SARIMAX # model SARIMA
from datetime import datetime, timedelta # konwerter to data
from pyspark.sql import functions as f # pozostałe funkcje
import statsmodels.api as sm #statystyki dopasowujące model
from pyspark.sql.window import Window
from pyspark.sql.functions import col, concat_ws, when, concat, date_format, to_date, format_number, round, last_day, expr,year, month, format_string


# COMMAND ----------

# MAGIC %fs
# MAGIC
# MAGIC ls /FileStore/tables

# COMMAND ----------

d = spark.read.format('csv').options(inferSchema = 'true', header = 'true', delimiter = ';').load('/FileStore/tables/danefinal.csv')

# COMMAND ----------

d.display(10)

# COMMAND ----------

d.filter("Data == '2021-07-01'").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Dodatkowe parametry - Dni tygodnia

# COMMAND ----------


d = d.withColumn('id_dnia_tygodnia', f.dayofweek(f.col('Data')))
d.display()


# COMMAND ----------

dni_tyg = spark.createDataFrame(
    [
    (2, "poniedziałek"),
    (3, "wtorek"),
    (4, "środa"),
    (5, "czwartek"),
    (6, "piątek"),
    (7, "sobota"),
    (1, "niedziela")    ],
    ("id","dzien")
)
dni_tyg.show()

# COMMAND ----------

d = d.join(dni_tyg, d.id_dnia_tygodnia == dni_tyg.id)
d.display()

# COMMAND ----------

d = d.withColumn('miesiac_rok', to_date(date_format('Data', 'yyyy-MM')))

# COMMAND ----------

d.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Kalendarz terminów

# COMMAND ----------


kalendarz = d.select("Data", "dzien_wolny") \
            .distinct() \
            .orderBy('Data') \
            .withColumnRenamed('Data', 'data1')

# COMMAND ----------

kalendarz.display()

# COMMAND ----------


# Tworzenie kolumny z oznaczeniem terminów, gdy termin jest w dzień pracujący
kalendarz = kalendarz.withColumn('TerminD', 
                   f.when((f.dayofmonth('data1') == 1) & (d['dzien_wolny'] == 'D'), 'T1')\
                   .when((f.dayofmonth('data1') == 1) & (d['dzien_wolny'] == 'DW'), 'T1x')\
                   .when((f.dayofmonth('data1') == 5) & (d['dzien_wolny'] == 'D'), 'T5')\
                   .when((f.dayofmonth('data1') == 5) & (d['dzien_wolny'] == 'DW'), 'T5x')\
                   .when((f.dayofmonth('data1') == 10) & (d['dzien_wolny'] == 'D'), 'T10')\
                   .when((f.dayofmonth('data1') == 10) & (d['dzien_wolny'] == 'DW'), 'T10x')\
                   .when((f.dayofmonth('data1') == 15) & (d['dzien_wolny'] == 'D'), 'T15')\
                   .when((f.dayofmonth('data1') == 15) & (d['dzien_wolny'] == 'DW'), 'T15x')\
                   .when((f.dayofmonth('data1') == 20) & (d['dzien_wolny'] == 'D'), 'T20')\
                   .when((f.dayofmonth('data1') == 20) & (d['dzien_wolny'] == 'DW'), 'T20x')\
                   .when((f.dayofmonth('data1') == 25) & (d['dzien_wolny'] == 'D'), 'T25')\
                   .when((f.dayofmonth('data1') == 25) & (d['dzien_wolny'] == 'DW'), 'T25x')\
                   .otherwise('')\
                  )



# COMMAND ----------

kalendarz.display()

# COMMAND ----------

# Określenie 'D-1', sprawdzenie czy poprzedni dzień jest pracujący, jeśli termin był wolnym
# Definicja kolumny TerminD
column_termin_d = f.lag('TerminD').over(Window.orderBy(f.col('data1').desc()))

# Określenie warunków dla kolumny TerminD-1
condition_t1 = (column_termin_d == 'T1x') & (f.col('dzien_wolny') == 'D')
condition_t1x = (column_termin_d == 'T1x') & (f.col('dzien_wolny') == 'DW')
condition_t5 = (column_termin_d == 'T5x') & (f.col('dzien_wolny') == 'D')
condition_t5x = (column_termin_d == 'T5x') & (f.col('dzien_wolny') == 'DW')
condition_t10 = (column_termin_d == 'T10x') & (f.col('dzien_wolny') == 'D')
condition_t10x = (column_termin_d == 'T10x') & (f.col('dzien_wolny') == 'DW')
condition_t15 = (column_termin_d == 'T15x') & (f.col('dzien_wolny') == 'D')
condition_t15x = (column_termin_d == 'T15x') & (f.col('dzien_wolny') == 'DW')
condition_t20 = (column_termin_d == 'T20x') & (f.col('dzien_wolny') == 'D')
condition_t20x = (column_termin_d == 'T20x') & (f.col('dzien_wolny') == 'DW')
condition_t25 = (column_termin_d == 'T25x') & (f.col('dzien_wolny') == 'D')
condition_t25x = (column_termin_d == 'T25x') & (f.col('dzien_wolny') == 'DW')

# Dodanie kolumny TerminD-1 na podstawie warunków
kalendarz = kalendarz.withColumn('TerminD-1',
                   f.when(condition_t1, 'T1')\
                   .when(condition_t1x, 'T1x')\
                   .when(condition_t5, 'T5')\
                   .when(condition_t5x, 'T5x')\
                   .when(condition_t10, 'T10')\
                   .when(condition_t10x, 'T10x')\
                   .when(condition_t15, 'T15')\
                   .when(condition_t15x, 'T15x')\
                   .when(condition_t20, 'T20')\
                   .when(condition_t20x, 'T20x')\
                   .when(condition_t25, 'T25')\
                   .when(condition_t25x, 'T25x')\
                   .otherwise('')\
                  )


# COMMAND ----------

kalendarz.display()

# COMMAND ----------

# Określenie 'D-2'i tak jeszcze 2 razy
# Definicja kolumny TerminD-1
column_termin_d = f.lag('TerminD-1').over(Window.orderBy(f.col('data1').desc()))

# Określenie warunków dla kolumny TerminD-1
condition_t1 = (column_termin_d == 'T1x') & (f.col('dzien_wolny') == 'D')
condition_t1x = (column_termin_d == 'T1x') & (f.col('dzien_wolny') == 'DW')
condition_t5 = (column_termin_d == 'T5x') & (f.col('dzien_wolny') == 'D')
condition_t5x = (column_termin_d == 'T5x') & (f.col('dzien_wolny') == 'DW')
condition_t10 = (column_termin_d == 'T10x') & (f.col('dzien_wolny') == 'D')
condition_t10x = (column_termin_d == 'T10x') & (f.col('dzien_wolny') == 'DW')
condition_t15 = (column_termin_d == 'T15x') & (f.col('dzien_wolny') == 'D')
condition_t15x = (column_termin_d == 'T15x') & (f.col('dzien_wolny') == 'DW')
condition_t20 = (column_termin_d == 'T20x') & (f.col('dzien_wolny') == 'D')
condition_t20x = (column_termin_d == 'T20x') & (f.col('dzien_wolny') == 'DW')
condition_t25 = (column_termin_d == 'T25x') & (f.col('dzien_wolny') == 'D')
condition_t25x = (column_termin_d == 'T25x') & (f.col('dzien_wolny') == 'DW')

# Dodanie kolumny TerminD-2 na podstawie warunków
kalendarz = kalendarz.withColumn('TerminD-2',
                   f.when(condition_t1, 'T1')\
                   .when(condition_t1x, 'T1x')\
                   .when(condition_t5, 'T5')\
                   .when(condition_t5x, 'T5x')\
                   .when(condition_t10, 'T10')\
                   .when(condition_t10x, 'T10x')\
                   .when(condition_t15, 'T15')\
                   .when(condition_t15x, 'T15x')\
                   .when(condition_t20, 'T20')\
                   .when(condition_t20x, 'T20x')\
                   .when(condition_t25, 'T25')\
                   .when(condition_t25x, 'T25x')\
                   .otherwise('')\
                  )

# Określenie 'D-3'
# Definicja kolumny TerminD-3
column_termin_d = f.lag('TerminD-2').over(Window.orderBy(f.col('data1').desc()))

# Określenie warunków dla kolumny TerminD-2
condition_t1 = (column_termin_d == 'T1x') & (f.col('dzien_wolny') == 'D')
condition_t1x = (column_termin_d == 'T1x') & (f.col('dzien_wolny') == 'DW')
condition_t5 = (column_termin_d == 'T5x') & (f.col('dzien_wolny') == 'D')
condition_t5x = (column_termin_d == 'T5x') & (f.col('dzien_wolny') == 'DW')
condition_t10 = (column_termin_d == 'T10x') & (f.col('dzien_wolny') == 'D')
condition_t10x = (column_termin_d == 'T10x') & (f.col('dzien_wolny') == 'DW')
condition_t15 = (column_termin_d == 'T15x') & (f.col('dzien_wolny') == 'D')
condition_t15x = (column_termin_d == 'T15x') & (f.col('dzien_wolny') == 'DW')
condition_t20 = (column_termin_d == 'T20x') & (f.col('dzien_wolny') == 'D')
condition_t20x = (column_termin_d == 'T20x') & (f.col('dzien_wolny') == 'DW')
condition_t25 = (column_termin_d == 'T25x') & (f.col('dzien_wolny') == 'D')
condition_t25x = (column_termin_d == 'T25x') & (f.col('dzien_wolny') == 'DW')

# Dodanie kolumny TerminD-1 na podstawie warunków
kalendarz = kalendarz.withColumn('TerminD-3',
                   f.when(condition_t1, 'T1')\
                   .when(condition_t1x, 'T1x')\
                   .when(condition_t5, 'T5')\
                   .when(condition_t5x, 'T5x')\
                   .when(condition_t10, 'T10')\
                   .when(condition_t10x, 'T10x')\
                   .when(condition_t15, 'T15')\
                   .when(condition_t15x, 'T15x')\
                   .when(condition_t20, 'T20')\
                   .when(condition_t20x, 'T20x')\
                   .when(condition_t25, 'T25')\
                   .when(condition_t25x, 'T25x')\
                   .otherwise('')\
                  )       

# COMMAND ----------

kalendarz.display()

# COMMAND ----------

# Tworzenie kolumny łączej 'wyk_term'
kalendarz = kalendarz.withColumn('wyk_term', 
                   concat_ws(';', 
                             when((col('TerminD') == 'T1') | 
                                  (col('TerminD-1') == 'T1') | 
                                  (col('TerminD-2') == 'T1') | 
                                  (col('TerminD-3') == 'T1'), 'T1'),
                             
                             when((col('TerminD') == 'T5') | 
                                  (col('TerminD-1') == 'T5') | 
                                  (col('TerminD-2') == 'T5') | 
                                  (col('TerminD-3') == 'T5'), 'T5'),
                             
                             when((col('TerminD') == 'T10') | 
                                  (col('TerminD-1') == 'T10') | 
                                  (col('TerminD-2') == 'T10') | 
                                  (col('TerminD-3') == 'T10'), 'T10'),
                             
                             when((col('TerminD') == 'T15') | 
                                  (col('TerminD-1') == 'T15') | 
                                  (col('TerminD-2') == 'T15') | 
                                  (col('TerminD-3') == 'T15'), 'T15'),
 
                             when((col('TerminD') == 'T20') | 
                                  (col('TerminD-1') == 'T20') | 
                                  (col('TerminD-2') == 'T20') | 
                                  (col('TerminD-3') == 'T20'), 'T20'),

                             when((col('TerminD') == 'T25') | 
                                  (col('TerminD-1') == 'T25') | 
                                  (col('TerminD-2') == 'T25') | 
                                  (col('TerminD-3') == 'T25'), 'T25'),                             

                            )
                  )

# COMMAND ----------

kalendarz.display()

# COMMAND ----------

# Przeddzień terminu - określenie tak samo jak dla T
kalendarz.drop('TerminD', 'TerminD-1', 'TerminD-2', 'TerminD-3')

# Określenie 'D-1'
# Definicja kolumny TerminD-1
column_termin_d = f.lag('wyk_term').over(Window.orderBy(f.col('data1').desc()))

# Określenie warunków dla kolumny TerminD-1
condition_t1D1 = (column_termin_d == 'T1') & (f.col('dzien_wolny') == 'D')
condition_t1D1x = (column_termin_d == 'T1') & (f.col('dzien_wolny') == 'DW')
condition_t5D1 = (column_termin_d == 'T5') & (f.col('dzien_wolny') == 'D')
condition_t5D1x = (column_termin_d == 'T5') & (f.col('dzien_wolny') == 'DW')
condition_t10D1 = (column_termin_d == 'T10') & (f.col('dzien_wolny') == 'D')
condition_t10D1x = (column_termin_d == 'T10') & (f.col('dzien_wolny') == 'DW')
condition_t15D1 = (column_termin_d == 'T15') & (f.col('dzien_wolny') == 'D')
condition_t15D1x = (column_termin_d == 'T15') & (f.col('dzien_wolny') == 'DW')
condition_t20D1 = (column_termin_d == 'T20') & (f.col('dzien_wolny') == 'D')
condition_t20D1x = (column_termin_d == 'T20') & (f.col('dzien_wolny') == 'DW')
condition_t25D1 = (column_termin_d == 'T25') & (f.col('dzien_wolny') == 'D')
condition_t25D1x = (column_termin_d == 'T25') & (f.col('dzien_wolny') == 'DW')

# Dodanie kolumny TerminD-1 na podstawie warunków
kalendarz = kalendarz.withColumn('TerminDD-1',
                   f.when(condition_t1D1, 'T1D-1')\
                   .when(condition_t1D1x, 'T1D-1x')\
                   .when(condition_t5D1, 'T5D-1')\
                   .when(condition_t5D1x, 'T5D-1x')\
                   .when(condition_t10D1, 'T10D-1')\
                   .when(condition_t10D1x, 'T10D-1x')\
                   .when(condition_t15D1, 'T15D-1')\
                   .when(condition_t15D1x, 'T15D-1x')\
                   .when(condition_t20D1, 'T20D-1')\
                   .when(condition_t20D1x, 'T20D-1x')\
                   .when(condition_t25D1, 'T25D-1')\
                   .when(condition_t25D1x, 'T25D-1x')\
                   .otherwise('')\
                  )

# Określenie 'D-2'
# Definicja kolumny TerminD-2
column_termin_d = f.lag('TerminDD-1').over(Window.orderBy(f.col('data1').desc()))

condition_t1D1 = (column_termin_d == 'T1D-1x') & (f.col('dzien_wolny') == 'D')
condition_t1D1x = (column_termin_d == 'T1D-1x') & (f.col('dzien_wolny') == 'DW')
condition_t5D1 = (column_termin_d == 'T5D-1x') & (f.col('dzien_wolny') == 'D')
condition_t5D1x = (column_termin_d == 'T5D-1x') & (f.col('dzien_wolny') == 'DW')
condition_t10D1 = (column_termin_d == 'T10D-1x') & (f.col('dzien_wolny') == 'D')
condition_t10D1x = (column_termin_d == 'T10D-1x') & (f.col('dzien_wolny') == 'DW')
condition_t15D1 = (column_termin_d == 'T15D-1x') & (f.col('dzien_wolny') == 'D')
condition_t15D1x = (column_termin_d == 'T15D-1x') & (f.col('dzien_wolny') == 'DW')
condition_t20D1 = (column_termin_d == 'T20D-1x') & (f.col('dzien_wolny') == 'D')
condition_t20D1x = (column_termin_d == 'T20D-1x') & (f.col('dzien_wolny') == 'DW')
condition_t25D1 = (column_termin_d == 'T25D-1x') & (f.col('dzien_wolny') == 'D')
condition_t25D1x = (column_termin_d == 'T25D-1x') & (f.col('dzien_wolny') == 'DW')

# Dodanie kolumny TerminD-2 na podstawie warunków
kalendarz = kalendarz.withColumn('TerminDD-2',
                   f.when(condition_t1D1, 'T1D-1')\
                   .when(condition_t1D1x, 'T1D-1x')\
                   .when(condition_t5D1, 'T5D-1')\
                   .when(condition_t5D1x, 'T5D-1x')\
                   .when(condition_t10D1, 'T10D-1')\
                   .when(condition_t10D1x, 'T10D-1x')\
                   .when(condition_t15D1, 'T15D-1')\
                   .when(condition_t15D1x, 'T15D-1x')\
                   .when(condition_t20D1, 'T20D-1')\
                   .when(condition_t20D1x, 'T20D-1x')\
                   .when(condition_t25D1, 'T25D-1')\
                   .when(condition_t25D1x, 'T25D-1x')\
                   .otherwise('')\
                  )

# Określenie 'D-3'
# Definicja kolumny TerminD-2
column_termin_d = f.lag('TerminDD-2').over(Window.orderBy(f.col('data1').desc()))

condition_t1D1 = (column_termin_d == 'T1D-1x') & (f.col('dzien_wolny') == 'D')
condition_t1D1x = (column_termin_d == 'T1D-1x') & (f.col('dzien_wolny') == 'DW')
condition_t5D1 = (column_termin_d == 'T5D-1x') & (f.col('dzien_wolny') == 'D')
condition_t5D1x = (column_termin_d == 'T5D-1x') & (f.col('dzien_wolny') == 'DW')
condition_t10D1 = (column_termin_d == 'T10D-1x') & (f.col('dzien_wolny') == 'D')
condition_t10D1x = (column_termin_d == 'T10D-1x') & (f.col('dzien_wolny') == 'DW')
condition_t15D1 = (column_termin_d == 'T15D-1x') & (f.col('dzien_wolny') == 'D')
condition_t15D1x = (column_termin_d == 'T15D-1x') & (f.col('dzien_wolny') == 'DW')
condition_t20D1 = (column_termin_d == 'T20D-1x') & (f.col('dzien_wolny') == 'D')
condition_t20D1x = (column_termin_d == 'T20D-1x') & (f.col('dzien_wolny') == 'DW')
condition_t25D1 = (column_termin_d == 'T25D-1x') & (f.col('dzien_wolny') == 'D')
condition_t25D1x = (column_termin_d == 'T25D-1x') & (f.col('dzien_wolny') == 'DW')

# Dodanie kolumny TerminD-2 na podstawie warunków
kalendarz = kalendarz.withColumn('TerminDD-3',
                   f.when(condition_t1D1, 'T1D-1')\
                   .when(condition_t1D1x, 'T1D-1x')\
                   .when(condition_t5D1, 'T5D-1')\
                   .when(condition_t5D1x, 'T5D-1x')\
                   .when(condition_t10D1, 'T10D-1')\
                   .when(condition_t10D1x, 'T10D-1x')\
                   .when(condition_t15D1, 'T15D-1')\
                   .when(condition_t15D1x, 'T15D-1x')\
                   .when(condition_t20D1, 'T20D-1')\
                   .when(condition_t20D1x, 'T20D-1x')\
                   .when(condition_t25D1, 'T25D-1')\
                   .when(condition_t25D1x, 'T25D-1x')\
                   .otherwise('')\
                  )
                                 

# COMMAND ----------

# Tworzenie kolumny łączej 'razemD_1'
kalendarz = kalendarz.withColumn('razemD_1', 
                   concat_ws(';', 
                             when((col('TerminDD-1') == 'T1D-1') | 
                                  (col('TerminDD-2') == 'T1D-1') | 
                                  (col('TerminDD-3') == 'T1D-1'), 'T1D-1'),
                             
                             when((col('TerminDD-1') == 'T5D-1') | 
                                  (col('TerminDD-2') == 'T5D-1') | 
                                  (col('TerminDD-3') == 'T5D-1'), 'T5D-1'),
                             
                             when((col('TerminDD-1') == 'T10D-1') | 
                                  (col('TerminDD-2') == 'T10D-1') | 
                                  (col('TerminDD-3') == 'T10D-1'), 'T10D-1'),
                             
                             when((col('TerminDD-1') == 'T15D-1') | 
                                  (col('TerminDD-2') == 'T15D-1') | 
                                  (col('TerminDD-3') == 'T15D-1'), 'T15D-1'),
 
                             when((col('TerminDD-1') == 'T20D-1') | 
                                  (col('TerminDD-2') == 'T20D-1') | 
                                  (col('TerminDD-3') == 'T20D-1'), 'T20D-1'),

                             when((col('TerminDD-1') == 'T25D-1') | 
                                  (col('TerminDD-2') == 'T25D-1') | 
                                  (col('TerminDD-3') == 'T25D-1'), 'T25D-1'),                             

                            )
                  )

# COMMAND ----------

# Tworzenie wyniku końcowego kalendarza
kalendarz = kalendarz.withColumn("Terminy", concat(kalendarz['wyk_term'], kalendarz['razemD_1']))
kalendarz = kalendarz.select('data1', 'Terminy').sort('data1')
kalendarz.show(10)

# COMMAND ----------

d = d.join(kalendarz, d.Data == kalendarz.data1)
d = d.drop('data1')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Badanie przepływów

# COMMAND ----------


#Podstawowe dane statystyczne
#Tabela
podst_st_df = d.groupBy('miesiac_rok')\
    .sum('zas_pl_R', 'zas_kz_R', 'nad_pl', 'nad_kz','zas_pl_W', 'zas_kz_W')\
    .sort('miesiac_rok')\
    .select('miesiac_rok',\
         format_number(round('sum(zas_pl_R)', 2), 2).alias('suma_mc_zasilki_do_placowek_R'),\
         format_number(round('sum(zas_kz_R)', 2), 2).alias('suma_mc_zasilki_dla_klientaZ_R'),\
         format_number(round('sum(nad_pl)', 2), 2).alias('suma_mc_nadmiary_z_placowek'),\
         format_number(round('sum(nad_kz)', 2), 2).alias('suma_mc_nadmiary_od_kientaZ'),\
         format_number(round('sum(zas_pl_W)', 2), 2).alias('suma_mc_zasilki_do_placowek_W'),\
         format_number(round('sum(zas_kz_W)', 2), 2).alias('suma_mc_zasilki_dla_klientaZ_W')\
         )

podst_st_df.display()

# Wykresy
podst_st1_df = d.groupBy('miesiac_rok')\
    .sum('zas_pl_R', 'zas_kz_R', 'nad_pl', 'nad_kz','zas_pl_W', 'zas_kz_W')\
    .sort('miesiac_rok')\
    .select('miesiac_rok',\
         round('sum(zas_pl_R)', 2).alias('suma_mc_zasilki_do_placowek_R'),\
         round('sum(zas_kz_R)', 2).alias('suma_mc_zasilki_dla_klientaZ_R'),\
         round('sum(nad_pl)', 2).alias('suma_mc_nadmiary_z_placowek'),\
         round('sum(nad_kz)', 2).alias('suma_mc_nadmiary_od_kientaZ'),\
         round('sum(zas_pl_W)', 2).alias('suma_mc_zasilki_do_placowek_W'),\
         round('sum(zas_kz_W)', 2).alias('suma_mc_zasilki_dla_klientaZ_W')\
         )

# Konwertowanie do DataFrame Pandas
podst_st1_pd = podst_st1_df.toPandas()

# Wykresy czasowe dla 6 parametrów
plt.figure(figsize=(14, 7))

plt.plot(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_zasilki_do_placowek_R'], label='suma_mc_zasilki_do_placowek_R')
plt.plot(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_zasilki_dla_klientaZ_R'], label='suma_mc_zasilki_dla_klientaZ_R')
plt.plot(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_nadmiary_z_placowek'], label='suma_mc_nadmiary_z_placowek')
plt.plot(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_nadmiary_od_kientaZ'], label='suma_mc_nadmiary_od_kientaZ')
plt.plot(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_zasilki_do_placowek_W'], label='suma_mc_zasilki_do_placowek_W')
plt.plot(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_zasilki_dla_klientaZ_W'], label='suma_mc_zasilki_dla_klientaZ_W')

plt.xlabel('Miesiąc i rok')
plt.ylabel('Suma')
plt.title('Sumy miesięczne poszczególnych parametrów obrotu gotówkowego')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)


# Wykresy czasowe dla 6 parametrów
plt.figure(figsize=(14, 28))

plt.subplot(6, 1, 1)
plt.bar(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_zasilki_do_placowek_R'], label='suma_mc_zasilki_do_placowek_R',width=20)
plt.xlabel('Miesiąc i rok')
plt.ylabel('Suma')
plt.title('Wykres dla suma_mc_zasilki_do_placowek_R')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.subplot(6, 1, 2)
plt.bar(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_zasilki_dla_klientaZ_R'], label='suma_mc_zasilki_dla_klientaZ_R',width=20)
plt.xlabel('Miesiąc i rok')
plt.ylabel('Suma')
plt.title('Wykres dla suma_mc_zasilki_dla_klientaZ_R')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.subplot(6, 1, 3)
plt.bar(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_nadmiary_z_placowek'], label='suma_mc_nadmiary_z_placowek',width=20)
plt.xlabel('Miesiąc i rok')
plt.ylabel('Suma')
plt.title('Wykres dla suma_mc_nadmiary_z_placowek')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.subplot(6, 1, 4)
plt.bar(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_nadmiary_od_kientaZ'], label='suma_mc_nadmiary_od_kientaZ',width=20)
plt.xlabel('Miesiąc i rok')
plt.ylabel('Suma')
plt.title('Wykres dla suma_mc_nadmiary_od_kientaZ')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.subplot(6, 1, 5)
plt.bar(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_zasilki_do_placowek_W'], label='suma_mc_zasilki_do_placowek_W',width=20)
plt.xlabel('Miesiąc i rok')
plt.ylabel('Suma')
plt.title('Wykres dla suma_mc_zasilki_do_placowek_W')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.subplot(6, 1, 6)
plt.bar(podst_st1_pd['miesiac_rok'], podst_st1_pd['suma_mc_zasilki_dla_klientaZ_W'], label='suma_mc_zasilki_dla_klientaZ_W',width=20)
plt.xlabel('Miesiąc i rok')
plt.ylabel('Suma')
plt.title('Wykres dla suma_mc_zasilki_dla_klientaZ_W')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# COMMAND ----------

#Tworzenie dataframe dla paramertu zasiłki dla nadmiary od klienta zewnętrznego
podst_st2a_df = d.groupBy('nazwa_kasy') \
                .pivot('miesiac_rok') \
                .sum("nad_kz")


# Tworzenie DataFrame z 'podst_st2_df'
podst_st2a_df_pd = podst_st2a_df.toPandas()

# Ustawienie 'nazwa_kasy' jako indeks
podst_st2a_df_pd.set_index('nazwa_kasy', inplace=True)

# Transpozycja DataFrame
podst_st2a_df_transposed = podst_st2a_df_pd.T

# Wybór interesujących kas
wybor_kasy = ['Portland', 'Detroit', 'Houston', 'Washington', 'Cleveland']

# Tworzenie wykresu
plt.figure(figsize=(8, 6))
for kas in wybor_kasy:
    plt.plot(podst_st2a_df_transposed.index, podst_st2a_df_transposed[kas], marker='o')

plt.title('Suma miesięczna nadmiarów od klienta zewętrznego dla wybranych kas')
plt.xlabel('Miesiące')
plt.ylabel('Suma')
plt.legend(wybor_kasy)
plt.xticks(rotation=45)  # Obrót oznaczeń osi x dla lepszej czytelności
plt.grid(True)
plt.tight_layout()  # Dopasowanie układu, aby uniknąć przecinania się etykiet osi
plt.show()



# COMMAND ----------

#Tworzenie dataframe dla paramertu zasiłki dla placówek poranne
podst_st2b_df = d.groupBy('nazwa_kasy') \
                .pivot('miesiac_rok') \
                .sum("zas_pl_R")


# Tworzenie DataFrame z 'podst_st2_df'
podst_st2b_df_pd = podst_st2b_df.toPandas()

# Ustawienie 'nazwa_kasy' jako indeks
podst_st2b_df_pd.set_index('nazwa_kasy', inplace=True)

# Transpozycja DataFrame
podst_st2b_df_transposed = podst_st2b_df_pd.T

# Wybór interesujących kas
wybor_kasy = ['Portland', 'Detroit', 'Houston', 'Washington', 'Cleveland']

# Tworzenie wykresu
plt.figure(figsize=(8, 6))
for kas in wybor_kasy:
    plt.plot(podst_st2b_df_transposed.index, podst_st2b_df_transposed[kas], marker='o')

plt.title('Suma miesięczna zasiłków dla placówek porannych dla wybranych kas')
plt.xlabel('Miesiące')
plt.ylabel('Suma')
plt.legend(wybor_kasy)
plt.xticks(rotation=45)  # Obrót oznaczeń osi x dla lepszej czytelności
plt.grid(True)
plt.tight_layout()  # Dopasowanie układu, aby uniknąć przecinania się etykiet osi
plt.show()



# COMMAND ----------



# Filtracja danych dla miesiąca  2022 roku
jeden_miesiac_danych = d.filter((f.year('Data') == 2022) & (f.month('Data') == 7))

# Pivotowanie danych nadmiary od klienta zew
podst_st3a_df = jeden_miesiac_danych.groupBy("nazwa_kasy").pivot("Data").agg(expr("first(nad_kz)"))
podst_st3a_df.display()

# Wybór pierwszych 5 kas
podst_st3a_df_5_kas = podst_st3a_df.limit(5)

# Tworzenie wykresu słupkowego dla pierwszych 5 kas
plt.figure(figsize=(12, 6))
for row in podst_st3a_df_5_kas.collect():  # Iteracja po wierszach DataFrame
    nazwa_kasy = row['nazwa_kasy']
    dane_kasy = list(row.asDict().values())[1:]  # Pomijamy pierwszą kolumnę 'nazwa_kasy'
    plt.plot(podst_st3a_df_5_kas.columns[1:], dane_kasy, label=nazwa_kasy)

plt.title('Dzienna wartość nadmiarów od kontrahenta w wybranych kasach w lipcu 2022 roku')
plt.xlabel('Dzień miesiąca')
plt.ylabel('Wartość nadmiarów')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()





# COMMAND ----------


# Filtracja danych dla miesiąca 2022 roku
jeden_miesiac_danych = d.filter((f.year('Data') == 2022) & (f.month('Data') == 7))

# Pivotowanie danych nadmiary od klienta zew
podst_st3b_df = jeden_miesiac_danych.groupBy("nazwa_kasy").pivot("Data").agg(expr("first(zas_kz_R)"))
podst_st3b_df.display()

# Wybór pierwszych 5 kas
podst_st3b_df_5_kas = podst_st3b_df.limit(5)

# Tworzenie wykresu słupkowego dla pierwszych 5 kas
plt.figure(figsize=(12, 6))
for row in podst_st3b_df_5_kas.collect():  # Iteracja po wierszach DataFrame
    nazwa_kasy = row['nazwa_kasy']
    dane_kasy = list(row.asDict().values())[1:]  # Pomijamy pierwszą kolumnę 'nazwa_kasy'
    plt.plot(podst_st3b_df_5_kas.columns[1:], dane_kasy, label=nazwa_kasy)

plt.title('Dzienna wartość zasiłków porannych dla kontrahenta w wybranych kasach w lipcu 2022 roku')
plt.xlabel('Dzień miesiąca')
plt.ylabel('Wartość zasiłków')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------


# Filtracja danych dla miesiąca 2022 roku
jeden_miesiac_danych = d.filter((f.year('Data') == 2022) & (f.month('Data') == 7))

# Pivotowanie danych nadmiary od klienta zew
podst_st3c_df = jeden_miesiac_danych.groupBy("nazwa_kasy").pivot("Data").agg(expr("first(zas_kz_W)"))
podst_st3c_df.display()

# Wybór pierwszych 5 kas
podst_st3c_df_5_kas = podst_st3c_df.limit(5)

# Tworzenie wykresu słupkowego dla pierwszych 5 kas
plt.figure(figsize=(12, 6))
for row in podst_st3c_df_5_kas.collect():  # Iteracja po wierszach DataFrame
    nazwa_kasy = row['nazwa_kasy']
    dane_kasy = list(row.asDict().values())[1:]  # Pomijamy pierwszą kolumnę 'nazwa_kasy'
    plt.plot(podst_st3c_df_5_kas.columns[1:], dane_kasy, label=nazwa_kasy)

plt.title('Dzienna wartość zasiłków wieczornych dla kontrahenta w wybranych kasach w lipcu 2022 roku')
plt.xlabel('Dzień miesiąca')
plt.ylabel('Wartość zasiłków')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------


# Filtracja danych dla miesiąca 2022 roku
jeden_miesiac_danych = d.filter((f.year('Data') == 2022) & (f.month('Data') == 7))

# Pivotowanie danych nadmiary od klienta zew
podst_st3d_df = jeden_miesiac_danych.groupBy("nazwa_kasy").pivot("Data").agg(expr("first(nad_pl)"))
podst_st3d_df.display()

# Wybór pierwszych 5 kas
podst_st3d_df_5_kas = podst_st3d_df.limit(5)

# Tworzenie wykresu słupkowego dla pierwszych 5 kas
plt.figure(figsize=(12, 6))
for row in podst_st3d_df_5_kas.collect():  # Iteracja po wierszach DataFrame
    nazwa_kasy = row['nazwa_kasy']
    dane_kasy = list(row.asDict().values())[1:]  # Pomijamy pierwszą kolumnę 'nazwa_kasy'
    plt.plot(podst_st3d_df_5_kas.columns[1:], dane_kasy, label=nazwa_kasy)

plt.title('Dzienna wartość nadmiarów z placówek w wybranych kasach w lipcu 2022 roku')
plt.xlabel('Dzień miesiąca')
plt.ylabel('Wartość nadmiarów')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------


# Filtracja danych dla miesiąca 2022 roku
jeden_miesiac_danych = d.filter((f.year('Data') == 2022) & (f.month('Data') == 7))

# Pivotowanie danych nadmiary od klienta zew
podst_st3e_df = jeden_miesiac_danych.groupBy("nazwa_kasy").pivot("Data").agg(expr("first(zas_pl_R)"))
podst_st3e_df.display()

# Wybór pierwszych 5 kas
podst_st3e_df_5_kas = podst_st3e_df.limit(5)

# Tworzenie wykresu słupkowego dla pierwszych 5 kas
plt.figure(figsize=(12, 6))
for row in podst_st3e_df_5_kas.collect():  # Iteracja po wierszach DataFrame
    nazwa_kasy = row['nazwa_kasy']
    dane_kasy = list(row.asDict().values())[1:]  # Pomijamy pierwszą kolumnę 'nazwa_kasy'
    plt.plot(podst_st3e_df_5_kas.columns[1:], dane_kasy, label=nazwa_kasy)

plt.title('Dzienna wartość zasiłków porannych dla placówek w wybranych kasach w lipcu 2022 roku')
plt.xlabel('Dzień miesiąca')
plt.ylabel('Wartość zasiłków')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------


# Filtracja danych dla miesiąca 2022 roku
jeden_miesiac_danych = d.filter((f.year('Data') == 2022) & (f.month('Data') == 7))

# Pivotowanie danych nadmiary od klienta zew
podst_st3f_df = jeden_miesiac_danych.groupBy("nazwa_kasy").pivot("Data").agg(expr("first(zas_pl_W)"))
podst_st3f_df.display()

# Wybór pierwszych 5 kas
podst_st3f_df_5_kas = podst_st3f_df.limit(5)

# Tworzenie wykresu słupkowego dla pierwszych 5 kas
plt.figure(figsize=(12, 6))
for row in podst_st3f_df_5_kas.collect():  # Iteracja po wierszach DataFrame
    nazwa_kasy = row['nazwa_kasy']
    dane_kasy = list(row.asDict().values())[1:]  # Pomijamy pierwszą kolumnę 'nazwa_kasy'
    plt.plot(podst_st3f_df_5_kas.columns[1:], dane_kasy, label=nazwa_kasy)

plt.title('Dzienna wartość zasiłków wieczornych dla placówek w wybranych kasach w lipcu 2022 roku')
plt.xlabel('Dzień miesiąca')
plt.ylabel('Wartość zasiłków')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------


podst_st4_df = d.filter(
    (f.year('Data') == 2022) & 
    (f.month('Data') == 7) & 
    (d['nazwa_kasy'] == 'Houston')
).select('Data', 'Terminy', 
          format_string('%,.0f', col('zas_pl_R').cast('float')).alias('zas_pl_R'),
          format_string('%,.0f', col('zas_pl_W').cast('float')).alias('zas_pl_W'),
          ) \
.sort('Data')
podst_st4_df.show()




# COMMAND ----------


# Ustawienie kolejności dni tygodnia
order_spec = f.when(f.col('dzien') == 'poniedziałek', 1) \
              .when(f.col('dzien') == 'wtorek', 2) \
              .when(f.col('dzien') == 'środa', 3) \
              .when(f.col('dzien') == 'czwartek', 4) \
              .when(f.col('dzien') == 'piątek', 5) \
              .when(f.col('dzien') == 'sobota', 6) \
              .when(f.col('dzien') == 'niedziela', 7) \
              .otherwise(99)  # Pozostałe dni tygodnia będą miały największy priorytet

podst_st5_df = d.filter((d['nazwa_kasy'] == 'Miami') & 
                        ((f.year('Data') == 2022) & ((f.month('Data') == 7) | (f.month('Data') == 8)))) \
                .orderBy(order_spec, f.col('Data')) \
                .select('Data', 'dzien', 'nad_kz')
                        #format_string('%,.0f', col('nad_kz').cast('float')).alias('nad_kz')) niestety psuje wykres

podst_st5_df.display()

# Utworzenie wykresu słupkowego
podst_st5_df = podst_st5_df.toPandas()
plt.figure(figsize=(12, 6))
plt.bar(podst_st5_df['dzien'], podst_st5_df['nad_kz'], width=0.6)
plt.xlabel('Dzień tygodnia')
plt.ylabel('Nadmiary')
plt.title('Średnie nadmiary w zależności od dnia tygodnia')
plt.show


# COMMAND ----------

podst_st6_df = d.filter(d['nazwa_kasy'] == 'Portland')\
    .select('Data', 'dzien', 'nad_kz')\
    .sort('Data')


plt.figure(figsize=(14, 6))
# Sporządzenie wykresu liniowego dla każdego dnia tygodnia
for day in ['poniedziałek', 'wtorek', 'środa', 'czwartek', 'piątek', 'sobota', 'niedziela']:
    # Filtracja danych dla danego dnia tygodnia
    day_data = podst_st6_df.filter(podst_st6_df['dzien'] == day).toPandas()
    
    # Wyodrębnienie dat i wartości 'nad_kz' dla danego dnia tygodnia
    dates = day_data['Data']
    nad_kz_values = day_data['nad_kz']
    
    # Sporządzenie wykresu liniowego
    plt.plot(dates, nad_kz_values, label=day)

# Dodanie legendy oraz tytułów osi
plt.xlabel('Data')
plt.ylabel('Wartość nad_kz')
plt.title('Przebieg wartości nad_kz wg dni tygodnia')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# Wyświetlenie wykresu
plt.show()



# COMMAND ----------

podst_st7_df = d.filter(d['nazwa_kasy'] == 'Portland')\
    .select('Data', 'Terminy', 'zas_pl_R')\
    .sort('Data')



plt.figure(figsize=(14, 6))
# Sporządzenie wykresu liniowego dla każdego dnia tygodnia
for term in ['T1', 'T1D-1', 'T5', 'T5D-1', 'T10', 'T10D-1', 'T15', 'T15D-1', 'T20', 'T20D-1', 'T25', 'T25D-1']:
    # Filtracja danych dla danego dnia tygodnia
    day_term = podst_st7_df.filter(podst_st7_df['Terminy'] == term).toPandas()
    
    # Wyodrębnienie dat i wartości 'zas_pl_R' dla danego dnia tygodnia
    dates = day_term['Data']
    zas_pl_R_values = day_term['zas_pl_R']
    
    # Sporządzenie wykresu liniowego
    plt.plot(dates, zas_pl_R_values, label=term)

# Dodanie legendy oraz tytułów osi
plt.xlabel('Data')
plt.ylabel('Wartość zasiłków porannych dla placówek')
plt.title('Przebieg wartości zasiłków dla placówek wg terminów')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Budowanie modelu

# COMMAND ----------

# Definiowanie zmiennych do sprwardzania innych konfiguracji
# Możliwe kasy:Cleveland, Filadelfia, Houston, Portland, Sacramento, Washington, Louisville, Detroit, Miami
# Możliwe parametry zas_pl_R, zas_kz_R, nad_pl, nad_kz, zas_pl_W, zas_kz_W

miasto = 'Houston'
parametr = 'nad_pl' 

#Sprawdzenie stacjonarności danych testem Dickey-Fullera
d1 = d.filter(d.nazwa_kasy == miasto).select('Data', parametr)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# Przekształcenie ramki danych
assembler = VectorAssembler(inputCols=[parametr], outputCol="features")
assembled_data = assembler.transform(d1).select("features")

# Wykonanie testu Dickey-Fullera
adf_test_result = Correlation.corr(assembled_data, "features", "pearson").collect()[0]


print("ADF Statistic:", adf_test_result['pearson(features)'])


# COMMAND ----------


d1pd = d1.toPandas()

# Konwersja kolumny 'Data' do typu daty
d1pd['Data'] = pd.to_datetime(d1pd['Data'])

# Sortowanie danych po dacie
d1pd.sort_values(by='Data', inplace=True)

# Różnicowanie danych
d1pd['parametr_diff'] = d1pd[parametr].diff()

# Wyświetlenie przekształconych danych
print(d1pd)


# COMMAND ----------

from statsmodels.tsa.stattools import adfuller

# Test Dickey-Fullera na kolumnie 'parametr_diff'
result = adfuller(d1pd['parametr_diff'].dropna())

# Wyświetlenie wyników testu
print('Test Dickey-Fullera:')
print('Statystyka testowa:', result[0])
print('P-value:', result[1])
print('Liczba lags:', result[2])
print('Liczba obserwacji:', result[3])
print('Wartości krytyczne:')
for key, value in result[4].items():
    print(f'   {key}: {value}')

# Interpretacja wyników
if result[1] <= 0.05:
    print("P-value <= 0.05. Odrzucamy hipotezę zerową. Szereg czasowy jest stacjonarny.")
else:
    print("P-value > 0.05. Nie ma podstaw do odrzucenia hipotezy zerowej. Szereg czasowy może być niestacjonarny.")


# COMMAND ----------


plt.figure(figsize=(10, 6))
plt.plot(d1pd['Data'], d1pd['parametr_diff'], marker='o', linestyle='-')
plt.title('Wykres szeregu czasowego parametr_diff')
plt.xlabel('Data')
plt.ylabel('parametr_diff')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(d1pd['parametr_diff'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram różnic parametr_diff')
plt.xlabel('Różnice parametr_diff')
plt.ylabel('Częstotliwość')
plt.grid(True)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Wykres autokorelacji
plot_acf(d1pd['parametr_diff'], lags=30)
plt.title('Wykres autokorelacji parametr_diff')
plt.xlabel('Opóźnienie')
plt.ylabel('Autokorelacja')
plt.show()

# Wykres autokorelacji częściowej
plot_pacf(d1pd['parametr_diff'], lags=30, method='ywm')
plt.title('Wykres autokorelacji częściowej parametr_diff')
plt.xlabel('Opóźnienie')
plt.ylabel('Autokorelacja częściowa')
plt.show()



# COMMAND ----------

d1pd.display() # mamy jeden null po różnicowaniu

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Podział danych na zbiór treningowy (90%) i testowy (10%)
train_data, test_data = train_test_split(d1pd, test_size=0.1, shuffle=False)
ostatni_wiersz = train_data.iloc[-1]
ostatnia_wartosc_parametru = ostatni_wiersz[parametr]
print(ostatnia_wartosc_parametru) # wartość początkowa do odzyskania po zróżnicowaniu
train_data.loc[train_data['Data'] == '2021-01-01', 'parametr_diff'] = 0 #usuwam pierwszy null po różnicowaniu

# COMMAND ----------

train_data.display()

# COMMAND ----------

print(train_data.dtypes)
print(train_data.isnull().sum())
train_data = train_data.sort_values(by='Data')

# COMMAND ----------

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Tworzenie modelu SARIMAX
order = (0, 1, 0)  # (p, d, q)
seasonal_order = (0, 1, 1, 7)  # (P, D, Q, S)


model = SARIMAX(train_data['parametr_diff'], order=order, seasonal_order=seasonal_order)

# Dopasowanie modelu do danych
results = model.fit()


# COMMAND ----------


# Prognozowanie zamodelowanych wartości
forecast = results.get_forecast(steps=len(test_data))
predicted_values = forecast.predicted_mean

# Obliczanie błędów prognozowania
residuals = test_data['parametr_diff'] - predicted_values

# Obliczanie metryk jakości modelu
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))
r_squared = 1 - (np.sum(residuals**2) / np.sum((test_data['parametr_diff'] - np.mean(test_data['parametr_diff']))**2))

# Test Ljunga-Box'a na błędach modelu
lags = 1
ljung_box_results = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=[lags])
p_value = ljung_box_results['lb_pvalue'][1]

# Wyświetlenie wyników
print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r_squared)
print("Ljung-Box test p-value:", p_value)



# COMMAND ----------

from pmdarima import auto_arima

# Automatyczne dopasowanie modelu SARIMA
auto_model = auto_arima(train_data['parametr_diff'], seasonal=True, m=7)

# Wyświetlenie najlepszych parametrów modelu
print(auto_model.summary())


# COMMAND ----------

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Tworzenie modelu SARIMAX po automatycznej paramertyzacji
order = (2, 0, 2)  # (p, d, q)
seasonal_order = (2, 0, 1, 7)  # (P, D, Q, S)


model = SARIMAX(train_data['parametr_diff'], order=order, seasonal_order=seasonal_order)

# Dopasowanie modelu do danych
results = model.fit()

# Prognozowanie zamodelowanych wartości
forecast = results.get_forecast(steps=len(test_data))
predicted_values = forecast.predicted_mean

# Obliczanie błędów prognozowania
residuals = test_data['parametr_diff'] - predicted_values

# Obliczanie metryk jakości modelu
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))
r_squared = 1 - (np.sum(residuals**2) / np.sum((test_data['parametr_diff'] - np.mean(test_data['parametr_diff']))**2))

# Test Ljunga-Box'a na błędach modelu
lags = 1
ljung_box_results = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=[lags])
p_value = ljung_box_results['lb_pvalue'][1]

# Wyświetlenie wyników
print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r_squared)
print("Ljung-Box test p-value:", p_value)


# COMMAND ----------


plt.figure(figsize=(12, 6))

# Wykres danych testowych
plt.plot(test_data.index, test_data['parametr_diff'], label='Dane testowe', color='blue')

# Wykres danych predykcyjnych
plt.plot(test_data.index, predicted_values, label='Dane predykcyjne', color='red')

# Dodanie etykiet i tytułu
plt.xlabel('Data')
plt.ylabel('Wartość parametru różnicowego')
plt.title('Porównanie danych predykcyjnych z danymi testowymi')
plt.legend()

# Wyświetlenie wykresu
plt.show()


# COMMAND ----------

# Odtwarzanie wartości testowych 
# Początkowa wartość zczytana z wcześniej określonej 'ostatnia_wartosc_parametru'
initial_value = ostatnia_wartosc_parametru

# Sekwencja różnic
diff_values = test_data['parametr_diff']

# Inicjalizacja listy z pierwszą wartością
original_values = [initial_value]

# Odtwarzanie oryginalnych wartości
for diff in diff_values:
    original_value = original_values[-1] + diff
    original_values.append(original_value)

original_values = original_values[1:] # usunięcie pierwszego dodatkowego wyrażenia
print(original_values)


# COMMAND ----------

# Odtwarzanie wartości predykcji
# Początkowa wartość
initial_value = ostatnia_wartosc_parametru

# Sekwencja różnic
diff_values = predicted_values

# Inicjalizacja listy z pierwszą wartością
original_values_predict = [initial_value]

# Odtwarzanie oryginalnych wartości
for diff in diff_values:
    original_value = original_values_predict[-1] + diff
    original_values_predict.append(original_value)
original_values_predict = original_values_predict[1:] # usunięcie pierwszego dodatkowego wyrażenia
print(original_values_predict)

# COMMAND ----------

plt.figure(figsize=(12, 6))

# Wykres danych testowych
plt.plot(test_data.index, original_values, label='Dane testowe', color='blue')

# Wykres danych predykcyjnych
plt.plot(test_data.index, original_values_predict, label='Dane predykcyjne', color='red')

# Dodanie etykiet i tytułu
plt.xlabel('Data')
plt.ylabel('Wartość parametru różnicowego')
plt.title('Porównanie danych predykcyjnych z danymi testowymi')
plt.legend()

# Wyświetlenie wykresu
plt.show()

# COMMAND ----------

porownanie_wynikow = pd.DataFrame({'Kolumna_1': original_values, 'Kolumna_2': original_values_predict})
porownanie_wynikow_sp = spark.createDataFrame(porownanie_wynikow)

# COMMAND ----------

porownanie_wynikow_sp = porownanie_wynikow_sp \
    .withColumnRenamed('Kolumna_1', 'dane_oryg') \
    .withColumnRenamed('Kolumna_2', 'dane_pred') \
    .withColumn('roznica', col('dane_oryg') - col('dane_pred')) \
    .withColumn('roznica', round(col('roznica'), 2)) \
    .withColumn('roznica_%', (col('dane_pred') / col('dane_oryg')) * 100) \
    .withColumn('roznica_%', round(col('roznica_%'), 2))

porownanie_wynikow_sp.show()

srednia_roznica = porownanie_wynikow_sp.agg(f.avg('roznica_%')).collect()[0][0]

print("Średnia różnica predykcji:", srednia_roznica)


# COMMAND ----------


