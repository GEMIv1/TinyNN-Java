package com.example.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class CSVDataReader {
private String filePath;
    private boolean hasHeader;
    private String delimiter;
    private List<String[]> data;
    private String[] headers;

    public CSVDataReader(String filePath) {
        this(filePath, true, ",");
    }

    public CSVDataReader(String filePath, boolean hasHeader) {
        this(filePath, hasHeader, ",");
    }

    public CSVDataReader(String filePath, boolean hasHeader, String delimiter) {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        this.filePath = filePath;
        this.hasHeader = hasHeader;
        this.delimiter = delimiter;
        this.data = new ArrayList<>();
    }

    public void loadData() throws IOException {
        data.clear();
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean isFirstLine = true;

            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] values = line.split(delimiter);
                
                for (int i = 0; i < values.length; i++) {
                    values[i] = values[i].trim();
                }

                if (isFirstLine && hasHeader) {
                    headers = values;
                    isFirstLine = false;
                } else {
                    data.add(values);
                    isFirstLine = false;
                }
            }
        }

        if (data.isEmpty()) {
            throw new IOException("No data found in CSV file");
        }
    }

    public String[][] getDataAsStringArray() {
        if (data.isEmpty()) {
            return new String[0][0];
        }
        return data.toArray(new String[0][]);
    }

    public double[][] getDataAsDoubleArray() {
        if (data.isEmpty()) {
            return new double[0][0];
        }

        int rows = data.size();
        int cols = data.get(0).length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            String[] row = data.get(i);
            for (int j = 0; j < cols; j++) {
                try {
                    result[i][j] = Double.parseDouble(row[j]);
                } catch (NumberFormatException e) {
                    throw new NumberFormatException(
                        "Cannot convert value '" + row[j] + "' at row " + i + ", column " + j + " to double"
                    );
                }
            }
        }

        return result;
    }

    public double[][] getColumnsAsDoubleArray(int... columnIndices) {
        if (data.isEmpty()) {
            return new double[0][0];
        }

        int rows = data.size();
        int cols = columnIndices.length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            String[] row = data.get(i);
            for (int j = 0; j < cols; j++) {
                int colIndex = columnIndices[j];
                if (colIndex < 0 || colIndex >= row.length) {
                    throw new IndexOutOfBoundsException("Column index " + colIndex + " is out of bounds");
                }
                try {
                    result[i][j] = Double.parseDouble(row[colIndex]);
                } catch (NumberFormatException e) {
                    throw new NumberFormatException(
                        "Cannot convert value '" + row[colIndex] + "' at row " + i + ", column " + colIndex + " to double"
                    );
                }
            }
        }

        return result;
    }

    public double[][] getColumnsByName(String... columnNames) {
        if (!hasHeader || headers == null) {
            throw new IllegalStateException("CSV file does not have headers");
        }

        int[] indices = new int[columnNames.length];
        for (int i = 0; i < columnNames.length; i++) {
            indices[i] = getColumnIndex(columnNames[i]);
            if (indices[i] == -1) {
                throw new IllegalArgumentException("Column '" + columnNames[i] + "' not found");
            }
        }

        return getColumnsAsDoubleArray(indices);
    }

    private int getColumnIndex(String columnName) {
        if (headers == null) {
            return -1;
        }
        for (int i = 0; i < headers.length; i++) {
            if (headers[i].equalsIgnoreCase(columnName)) {
                return i;
            }
        }
        return -1;
    }

    public String[] getHeaders() {
        return headers;
    }

    public int getRowCount() {
        return data.size();
    }

    public int getColumnCount() {
        if (data.isEmpty()) {
            return 0;
        }
        return data.get(0).length;
    }

    public String[] getRow(int rowIndex) {
        if (rowIndex < 0 || rowIndex >= data.size()) {
            throw new IndexOutOfBoundsException("Row index " + rowIndex + " is out of bounds");
        }
        return data.get(rowIndex);
    }
    
    public void printSummary() {
        System.out.println("CSV File: " + filePath);
        System.out.println("Rows: " + getRowCount());
        System.out.println("Columns: " + getColumnCount());
        if (hasHeader && headers != null) {
            System.out.println("Headers: " + String.join(", ", headers));
        }
    }
}
