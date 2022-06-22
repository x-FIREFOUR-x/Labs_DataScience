USE DB_Car_Star

CREATE TABLE FactTable
(
	IdFactTable int IDENTITY(1,1),
	IdCar int NULL FOREIGN KEY REFERENCES CarDim(IdCar),
	IdYear int NULL FOREIGN KEY REFERENCES YearDim(IdYear),
	IdBody int NULL FOREIGN KEY REFERENCES BodyDim(IdBody),
	IdFuel int NULL FOREIGN KEY REFERENCES FuelDim(IdFuel),
	IdTransmission int NULL FOREIGN KEY REFERENCES TransmissionDim(IdTransmission),
	IdDrive int NULL FOREIGN KEY REFERENCES DriveDim(IdDrive),

	Price nvarchar(100) NULL,
	MSRP nvarchar(100) NULL,
	EngineHP int NULL,
	EngineVolume nvarchar(100) NULL,
	EmissionCO2 int NULL,
	EmissionCO int NULL,
	EmissionNOx int NULL,
	NoiseLevel nvarchar(30) NULL,
	Popularity int NULL
)

DROP TABLE FactTable

SELECT * FROM FactTable

SELECT * FROM CarDim 

SELECT * FROM TransmissionDim

SELECT * FROM YearDim

SELECT * FROM BodyDim

SELECT * FROM FuelDim


SELECT  Brand, Model, [Year], Body, Fuel, Transmission, Drive
		Price, MSRP, EngineHP, EngineVolume, EmissionCO2, EmissionCO, EmissionNOx, NoiseLevel, Popularity
FROM FactTable
FULL JOIN CarDim on FactTable.IdCar = CarDim.IdCar
FULL JOIN YearDim ON FactTable.[IdYear] = YearDim.[IdYear]
FULL JOIN BodyDim ON FactTable.IdBody = BodyDim.IdBody
FULL JOIN FuelDim ON FactTable.IdFuel = FuelDim.IdFuel
FULL JOIN TransmissionDim ON FactTable.IdTransmission = TransmissionDim.IdTransmission
FULL JOIN DriveDim ON FactTable.IdDrive = DriveDim.IdDrive

SELECT * FROM CarSales

Select * FROM CarInform

SELECT * FROM CarEmissions