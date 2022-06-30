USE DB_Car_Star

CREATE TABLE [BodyDim]
(	IdBody int IDENTITY(1,1) PRIMARY KEY, 
	[Body] nvarchar(30) UNIQUE
)

INSERT INTO	[BodyDim]
SELECT [Body]
FROM CarSales
UNION
SELECT [Body]
FROM CarInform
Group by [Body]
ORDER BY [Body]

SELECT * FROM [BodyDim]