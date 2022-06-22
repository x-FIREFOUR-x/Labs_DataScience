USE DB_Car_Star

CREATE TABLE [YearDim]
(	IdYear int IDENTITY(1,1) PRIMARY KEY, 
	[Year] int UNIQUE
)

INSERT INTO	[YearDim]
SELECT [Year]
FROM CarSales
UNION
SELECT [Year]
FROM CarInform
UNION
SELECT [Year]
FROM CarEmissions
Group by [Year]
ORDER BY [Year]

SELECT * FROM [YearDim]