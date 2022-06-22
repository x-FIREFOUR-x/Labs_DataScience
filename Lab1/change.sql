USE DB_Car_Star

--UPDATE [CarEmissions]
--SET [Year] = (SELECT (ABS(CHECKSUM(NEWID()))% 17) + 2000) 

--Alter Table CarEmissions
--ADD [Year] nvarchar(4)

--UPDATE [CarEmissions]
--SET Model =( SELECT TOP(1) * FROM string_split(Model, '-'))

--UPDATE [CarEmissions]
--SET Model = (SELECT REPLACE(Model, 'A ', 'Class-A')) 

--UPDATE [CarEmissions]
--SET Model = 'Class-V' 
--WHERE Model = 'V' and Brand = 'MERCEDES-BENZ'

--UPDATE [CarEmissions]
--SET Fuel = (SELECT REPLACE(Fuel, 'Petrol Hybrid', 'Electricity / Petrol')) 

--UPDATE [CarSales]
--SET Body = (SELECT REPLACE(Body, 'other', 'Other')) 

--SELECT * FROM CarSales







