USE DB_Car_Star

SELECT  Price, Mileage, EngineVolume, Drive, 
		CarSales.Brand, CarSales.Model, CarSales.[Year], CarSales._Fuel, CarSales.Body,
		IdCar, IdYear, IdFuel, IdBody
FROM CarSales
JOIN CarDim on CarSales.Brand = CarDim.Brand and CarSales.Model = CarDim.Model
JOIN YearDim ON CarSales.[Year] = YearDim.[Year]
JOIN FuelDim ON CarSales._Fuel = FuelDim.Fuel
JOIN BodyDim ON CarSales.Body = BodyDim.Body


Select EngineHP, EngineCylinders, Popularity, MSRP,
	   CarInform.Brand, CarInform.Model, CarInform.[Year], CarInform.TransmissionType, CarInform.Body,
	   IdCar, IdYear, IdTransmission, IdBody
FROM CarInform
JOIN CarDim on CarInform.Brand = CarDim.Brand and CarInform.Model = CarDim.Model
JOIN YearDim ON CarInform.[Year] = YearDim.[Year]
JOIN TransmissionDim ON CarInform.TransmissionType = TransmissionDim.Transmission
JOIN BodyDim ON CarInform.Body = BodyDim.Body

SELECT [CO2(g/km)], [NoiseLeveldB(A)], [EmissionsCO(mg/km)], [EmissionsNOx(mg/km)],
	   CarEmissions.Brand,CarEmissions.Model, CarEmissions.[Year], CarEmissions.Fuel,
	   IdCar, IdYear, IdFuel
FROM CarEmissions
JOIN CarDim on CarEmissions.Brand = CarDim.Brand and CarEmissions.Model = CarDim.Model
JOIN YearDim ON CarEmissions.[Year] = YearDim.[Year]
JOIN FuelDim ON CarEmissions.Fuel = FuelDim.Fuel





SELECT  IdCar, IdYear, IdBody, IdFuel, NULL, 
		Price, NULL, NULL, EngineVolume , NULL, NULL 
FROM CarSales
JOIN CarDim on CarSales.Brand = CarDim.Brand and CarSales.Model = CarDim.Model
JOIN YearDim ON CarSales.[Year] = YearDim.[Year]
JOIN FuelDim ON CarSales._Fuel = FuelDim.Fuel
JOIN BodyDim ON CarSales.Body = BodyDim.Body


Select IdCar, IdYear, IdBody, NULL , IdTransmission,
	   NULL, MSRP, EngineHP, NULL, NULL, NULL
FROM CarInform
JOIN CarDim on CarInform.Brand = CarDim.Brand and CarInform.Model = CarDim.Model
JOIN YearDim ON CarInform.[Year] = YearDim.[Year]
JOIN TransmissionDim ON CarInform.TransmissionType = TransmissionDim.Transmission
JOIN BodyDim ON CarInform.Body = BodyDim.Body


SELECT IdCar, IdYear, NULL, IdFuel, NULL,
		NULL, NULL, NULL, NULL, [CO2(g/km)], [EmissionsCO(mg/km)]
FROM CarEmissions
JOIN CarDim on CarEmissions.Brand = CarDim.Brand and CarEmissions.Model = CarDim.Model
JOIN YearDim ON CarEmissions.[Year] = YearDim.[Year]
JOIN FuelDim ON CarEmissions.Fuel = FuelDim.Fuel