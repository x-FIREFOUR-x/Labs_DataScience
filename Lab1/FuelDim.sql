USE DB_Car_Star

CREATE TABLE [FuelDim]
(	IdFuel int IDENTITY(1,1) PRIMARY KEY, 
	[Fuel] nvarchar(30) UNIQUE
)

INSERT INTO	[FuelDim]
SELECT [_Fuel]
FROM CarSales
UNION
SELECT [Fuel]
FROM CarEmissions
--Group by [Fuel]
ORDER BY [_Fuel]

SELECT * FROM [FuelDim]