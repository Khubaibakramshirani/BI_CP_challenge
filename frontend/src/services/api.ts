// src/services/api.ts
import axios from 'axios';
import { BASE_API_URL } from '../constants';  // Import the constant


const apiClient = axios.create({
    baseURL: BASE_API_URL, // Backend API base URL
    headers: {
        'Content-Type': 'application/json',
    },
});

export const submitQuery = async (question: string) => {
    try {
        const response = await apiClient.post('/query', { question });
        return response.data;
    } catch (error) {
        throw new Error('Failed to fetch the response. Please try again later.');
    }
};

