import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Sprint": ["Sprint 1", "Sprint 2", "Sprint 3", "Sprint 4", "Sprint 5", "Sprint 6"],
    "Tanmayi": [4, 5, 6, 5, 6, 6],
    "Sameera": [6, 5, 5, 5, 5, 6]
}

df = pd.DataFrame(data)

df["Team_Average"] = df[["Tanmayi", "Sameera"]].mean(axis=1)

print("Historical Velocity Table with Team Average:")
print(df[["Sprint", "Tanmayi", "Sameera", "Team_Average"]])

plt.figure(figsize=(10,6))

plt.plot(df["Sprint"], df["Tanmayi"], marker='o', label="Tanmayi")
plt.plot(df["Sprint"], df["Sameera"], marker='o', label="Sameera")

plt.plot(df["Sprint"], df["Team_Average"], marker='x', label="Team Average")

plt.title("Velocity per Sprint")
plt.xlabel("Sprint")
plt.ylabel("points")
plt.ylim(0, max(df[["Tanmayi","Sameera"]].max().max()+2, 8))
plt.legend()
plt.grid(True)
plt.show()
