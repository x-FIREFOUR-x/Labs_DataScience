USE DB_Car_Star


	--3 співпало
INSERT INTO	[FactTable]
SELECT IdCar, IdYear, IdBody, IdFuel, IdTransmission, IdDrive,
	   Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)], [EmissionsNOx(mg/km)],
	   [NoiseLeveldB(A)], Popularity
FROM (	 
		SELECT	CarSales.Brand, CarSales.Model, CarSales.[Year], CarSales.Body, CarSales._Fuel, CarInform.TransmissionType, CarSales.Drive,
				Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)] , [EmissionsNOx(mg/km)],
				[NoiseLeveldB(A)], Popularity
		FROM CarSales
		FULL JOIN CarInform ON CarSales.Brand = CarInform.Brand and CarSales.Model = CarInform.Model and CarSales.[Year] = CarInform.[Year]
		FULL JOIN CarEmissions ON CarSales.Brand = CarEmissions.Brand and CarSales.Model = CarEmissions.Model and CarSales.[Year] = CarEmissions.[Year]
		WHERE CarSales.Brand IS NOT NULL and CarInform.Brand IS NOT NULL and CarEmissions.Brand IS NOT NULL
) AS table1
JOIN CarDim on table1.Brand = CarDim.Brand and table1.Model = CarDim.Model
JOIN YearDim ON table1.[Year] = YearDim.[Year]
JOIN TransmissionDim ON table1.TransmissionType = TransmissionDim.Transmission
JOIN BodyDim ON table1.Body = BodyDim.Body
JOIN FuelDim ON table1._Fuel = FuelDim.Fuel
JOIN DriveDim ON table1.Drive = DriveDim.Drive


			-- Salse(1) Null
SELECT IdCar, IdYear, IdBody, IdFuel, IdTransmission, NULL,
	   Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)], [EmissionsNOx(mg/km)],
	   [NoiseLeveldB(A)], Popularity
FROM (	 
		SELECT	CarInform.Brand, CarInform.Model, CarInform.[Year], CarInform.Body, CarEmissions.Fuel, CarInform.TransmissionType,CarSales.Drive,
				Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)] , [EmissionsNOx(mg/km)],
				[NoiseLeveldB(A)], Popularity
		FROM CarSales
		FULL JOIN CarInform ON CarSales.Brand = CarInform.Brand and CarSales.Model = CarInform.Model and CarSales.[Year] = CarInform.[Year]
		FULL JOIN CarEmissions ON CarSales.Brand = CarEmissions.Brand and CarSales.Model = CarEmissions.Model and CarSales.[Year] = CarEmissions.[Year]
		WHERE CarSales.Brand IS NULL and CarInform.Brand IS NOT NULL and CarEmissions.Brand IS NOT NULL
) AS table2
JOIN CarDim on table2.Brand = CarDim.Brand and table2.Model = CarDim.Model
JOIN YearDim ON table2.[Year] = YearDim.[Year]
JOIN TransmissionDim ON table2.TransmissionType = TransmissionDim.Transmission
JOIN BodyDim ON table2.Body = BodyDim.Body
JOIN FuelDim ON table2.Fuel = FuelDim.Fuel
--JOIN DriveDim ON table2.Drive = DriveDim.Drive

			-- Inform(2) NULL
INSERT INTO	[FactTable]
SELECT IdCar, IdYear, IdBody, IdFuel, NULL,IdDrive,
	   Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)], [EmissionsNOx(mg/km)],
	   [NoiseLeveldB(A)], Popularity
FROM (	 
		SELECT	CarSales.Brand, CarSales.Model, CarSales.[Year], CarSales.Body, CarSales._Fuel, CarInform.TransmissionType,CarSales.Drive,
				Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)] , [EmissionsNOx(mg/km)],
				[NoiseLeveldB(A)], Popularity
		FROM CarSales
		FULL JOIN CarInform ON CarSales.Brand = CarInform.Brand and CarSales.Model = CarInform.Model and CarSales.[Year] = CarInform.[Year]
		FULL JOIN CarEmissions ON CarSales.Brand = CarEmissions.Brand and CarSales.Model = CarEmissions.Model and CarSales.[Year] = CarEmissions.[Year]
		WHERE CarSales.Brand IS NOT NULL and CarInform.Brand IS NULL and CarEmissions.Brand IS NOT NULL
) AS table3
JOIN CarDim on table3.Brand = CarDim.Brand and table3.Model = CarDim.Model
JOIN YearDim ON table3.[Year] = YearDim.[Year]
--JOIN TransmissionDim ON table3.TransmissionType = TransmissionDim.Transmission
JOIN BodyDim ON table3.Body = BodyDim.Body
JOIN FuelDim ON table3._Fuel = FuelDim.Fuel
JOIN DriveDim ON table3.Drive = DriveDim.Drive

			--Emissions(3) Null
INSERT INTO	[FactTable]
SELECT IdCar, IdYear, IdBody, IdFuel, IdTransmission,IdDrive,
	   Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)], [EmissionsNOx(mg/km)],
	   [NoiseLeveldB(A)], Popularity 
FROM (	 
		SELECT	CarSales.Brand, CarSales.Model, CarSales.[Year], CarSales.Body, CarSales._Fuel, CarInform.TransmissionType, CarSales.Drive,
				Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)] , [EmissionsNOx(mg/km)],
				[NoiseLeveldB(A)], Popularity
		FROM CarSales
		FULL JOIN CarInform ON CarSales.Brand = CarInform.Brand and CarSales.Model = CarInform.Model and CarSales.[Year] = CarInform.[Year]
		FULL JOIN CarEmissions ON CarSales.Brand = CarEmissions.Brand and CarSales.Model = CarEmissions.Model and CarSales.[Year] = CarEmissions.[Year]
		WHERE CarSales.Brand IS NOT NULL and CarInform.Brand IS NOT NULL and CarEmissions.Brand IS NULL
) AS table4
JOIN CarDim on table4.Brand = CarDim.Brand and table4.Model = CarDim.Model
JOIN YearDim ON table4.[Year] = YearDim.[Year]
JOIN TransmissionDim ON table4.TransmissionType = TransmissionDim.Transmission
JOIN BodyDim ON table4.Body = BodyDim.Body
JOIN FuelDim ON table4._Fuel = FuelDim.Fuel
JOIN DriveDim ON table4.Drive = DriveDim.Drive


		--Sales i Inform (1, 2) NULL
INSERT INTO	[FactTable]
SELECT IdCar, IdYear, NULL, IdFuel, NULL, NULL,
	   Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)], [EmissionsNOx(mg/km)],
	   [NoiseLeveldB(A)], Popularity
FROM (	 
		SELECT	CarEmissions.Brand, CarEmissions.Model, CarEmissions.[Year], NULL AS Body, CarEmissions.Fuel, NULL AS Transmissions,CarSales.Drive,
				Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)] , [EmissionsNOx(mg/km)],
				[NoiseLeveldB(A)], Popularity
		FROM CarSales
		FULL JOIN CarInform ON CarSales.Brand = CarInform.Brand and CarSales.Model = CarInform.Model and CarSales.[Year] = CarInform.[Year]
		FULL JOIN CarEmissions ON CarSales.Brand = CarEmissions.Brand and CarSales.Model = CarEmissions.Model and CarSales.[Year] = CarEmissions.[Year]
		WHERE CarSales.Brand IS NULL and CarInform.Brand IS NULL and CarEmissions.Brand IS NOT NULL
) AS table5
JOIN CarDim on table5.Brand = CarDim.Brand and table5.Model = CarDim.Model
JOIN YearDim ON table5.[Year] = YearDim.[Year]
--JOIN TransmissionDim ON table5.TransmissionType = TransmissionDim.Transmission
--JOIN BodyDim ON table5.Body = BodyDim.Body
JOIN FuelDim ON table5.Fuel = FuelDim.Fuel
--JOIN DriveDim ON table5.Drive = DriveDim.Drive


		-- Inform i Emisions (2 3) NULL
INSERT INTO	[FactTable]
SELECT IdCar, IdYear, IdBody, IdFuel, NULL AS Transmission,IdDrive,
	   Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)], [EmissionsNOx(mg/km)],
	   [NoiseLeveldB(A)], Popularity
FROM (	 
		SELECT	CarSales.Brand, CarSales.Model, CarSales.[Year], CarSales.Body, CarSales._Fuel, CarInform.TransmissionType, CarSales.Drive,
				Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)] , [EmissionsNOx(mg/km)],
				[NoiseLeveldB(A)], Popularity
		FROM CarSales
		FULL JOIN CarInform ON CarSales.Brand = CarInform.Brand and CarSales.Model = CarInform.Model and CarSales.[Year] = CarInform.[Year]
		FULL JOIN CarEmissions ON CarSales.Brand = CarEmissions.Brand and CarSales.Model = CarEmissions.Model and CarSales.[Year] = CarEmissions.[Year]
		WHERE CarSales.Brand IS NOT NULL and CarInform.Brand IS NULL and CarEmissions.Brand IS NULL
) AS table6
JOIN CarDim on table6.Brand = CarDim.Brand and table6.Model = CarDim.Model
JOIN YearDim ON table6.[Year] = YearDim.[Year]
--JOIN TransmissionDim ON table6.TransmissionType = TransmissionDim.Transmission
JOIN BodyDim ON table6.Body = BodyDim.Body
JOIN FuelDim ON table6._Fuel = FuelDim.Fuel
JOIN DriveDim ON table6.Drive = DriveDim.Drive

		--Salse i Emisions (1 3) NULL
INSERT INTO	[FactTable]
SELECT IdCar, IdYear, IdBody, NULL AS Fuel, IdTransmission,NULL,
	   Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)], [EmissionsNOx(mg/km)],
	   [NoiseLeveldB(A)], Popularity
FROM (	 
		SELECT	CarInform.Brand, CarInform.Model, CarInform.[Year], CarInform.Body, NULL AS Fuel, CarInform.TransmissionType, CarSales.Drive,
				Price,  MSRP, EngineHP, EngineVolume , [CO2(g/km)], [EmissionsCO(mg/km)] , [EmissionsNOx(mg/km)],
				[NoiseLeveldB(A)], Popularity
		FROM CarSales
		FULL JOIN CarInform ON CarSales.Brand = CarInform.Brand and CarSales.Model = CarInform.Model and CarSales.[Year] = CarInform.[Year]
		FULL JOIN CarEmissions ON CarSales.Brand = CarEmissions.Brand and CarSales.Model = CarEmissions.Model and CarSales.[Year] = CarEmissions.[Year]
		WHERE CarSales.Brand IS NULL and CarInform.Brand IS NOT NULL and CarEmissions.Brand IS NULL
) AS table7
JOIN CarDim on table7.Brand = CarDim.Brand and table7.Model = CarDim.Model
JOIN YearDim ON table7.[Year] = YearDim.[Year]
JOIN TransmissionDim ON table7.TransmissionType = TransmissionDim.Transmission
JOIN BodyDim ON table7.Body = BodyDim.Body
--JOIN FuelDim ON table7.Fuel = FuelDim.Fuel
--JOIN DriveDim ON table7.Drive = DriveDim.Drive