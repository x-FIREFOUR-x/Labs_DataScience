USE DB_Car_Star

CREATE TABLE [CarDim]
(	IdCar int IDENTITY(1,1) PRIMARY KEY, 
	[Brand] nvarchar(30) ,
	[Model] nvarchar(100) 
)

DROP TABLE [CarDim]

INSERT INTO	[CarDim]
SELECT [Brand], [Model]
FROM CarSales
UNION
SELECT [Brand], [Model]
FROM CarInform
UNION
SELECT [Brand], [Model]
FROM CarEmissions
Group by [Brand], [Model]
ORDER BY [Brand], [Model]

SELECT * FROM [CarDim]