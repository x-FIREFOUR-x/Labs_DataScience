USE DB_Car_Star

CREATE TABLE [DriveDim]
(	IdDrive int IDENTITY(1,1) PRIMARY KEY, 
	[Drive] nvarchar(30) 
)

DROP TABLE [CarDim]

INSERT INTO	[DriveDim]
SELECT Drive
FROM CarSales
Group by Drive
ORDER BY Drive

SELECT * FROM [DriveDim]