USE DB_Car_Star

CREATE TABLE [TransmissionDim]
(	IdTransmission int IDENTITY(1,1) PRIMARY KEY, 
	[Transmission] nvarchar(30) UNIQUE
)

DROP TABLE [TransmisionDim]

INSERT INTO	[TransmissionDim]
SELECT [TransmissionType]
FROM CarInform
Group by [TransmissionType]
ORDER BY [TransmissionType]

SELECT * FROM [TransmissionDim]